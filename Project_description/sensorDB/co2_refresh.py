"""Versioned, resumable full refresh of the 53 workbook LEO West CO2 channels.

No old archives are read. A manifest is the commit record for every raw part.
Oracle NUMBER/FLOAT are fetched as Decimal and preserved in separate Parquet
parts (each part has its own exact decimal schema). Never run concurrent writers.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT/'Project_description/sensorDB/Row_data/LEO_West_full'
WORKBOOK = ROOT/'Sensors_Description/variables_schema.xlsx'
PART_ROWS = 50000
MAX_BLOCK_ROWS = 1000000
RETRIES = 3
LOG = logging.getLogger('co2-refresh')


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def sync_dir(path):
    fd=os.open(path,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def atomic_json(path,data):
    path=Path(path); temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as stream:
        json.dump(data,stream,ensure_ascii=False,indent=2,default=str)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path);sync_dir(path.parent)


def workbook_channels():
    sys.path.insert(0,str(ROOT/'scripts'))
    from update_co2_sheet import selected_records,load_workbook
    book=load_workbook(WORKBOOK,data_only=True)
    try:
        records=selected_records(book['CO2'],{'LEO West'})
        channels=[dict(row=row,table=r['N'].upper(),sensorid=int(r['L']),variableid=int(r['AD']),
                       sensorcode=r['M'],kind=r['K'],x=float(r['C']),y=float(r['D']),
                       z=float(r['E']),units=r['AG']) for row,r in records
                  if r['K']=='GMM222' or (r['K']=='LI-COR' and r['E']==.25)]
    finally: book.close()
    if Counter(c['kind'] for c in channels)!=Counter({'GMM222':48,'LI-COR':5}):
        raise ValueError('Expected 48 GMM222 + 5 LI-COR workbook channels')
    if len({(c['x'],c['y']) for c in channels if c['kind']=='GMM222'})!=16:
        raise ValueError('Expected 16 basalt verticals')
    return channels


class Oracle:
    def __init__(self): self.conn=None
    def close(self):
        if self.conn:
            try:self.conn.close()
            except Exception:pass
        self.conn=None
    def connection(self):
        if self.conn is None:
            sys.path.insert(0,str(ROOT/'scripts'))
            import oracledb
            from update_co2_sheet import connect
            oracledb.defaults.fetch_decimals=True
            self.conn=connect();self.conn.call_timeout=120000
        return self.conn
    def select(self,sql,params=None):
        with self.connection().cursor() as cur:
            cur.execute(sql,params or {});return cur.fetchall()
    def stream(self,sql,params):
        with self.connection().cursor() as cur:
            cur.arraysize=10000;cur.prefetchrows=10000
            cur.execute(sql,params)
            while True:
                rows=cur.fetchmany(PART_ROWS)
                if not rows:break
                yield rows


def channel_where(channels):
    variableids={c['variableid'] for c in channels}
    if len(variableids)!=1:raise ValueError('Mixed variable IDs')
    return 'sensorid IN ('+','.join(str(c['sensorid']) for c in channels)+') AND variableid='+str(next(iter(variableids)))


def inspect_metadata(db,channels):
    ids=','.join(str(c['sensorid']) for c in channels)
    rows=db.select(f'''SELECT s.sensorid,s.sensorcode,s.datatablename,l.localx,l.localy,l.localz,
        l.boxx,l.boxy,l.boxz,l.dlevel FROM leo_west.sensors s JOIN leo_west.locations l
        ON s.locationid=l.locationid WHERE s.sensorid IN ({ids})''')
    sensors={int(r[0]):r for r in rows}
    if len(rows)!=len(channels):raise ValueError('Oracle sensor identity is not one-to-one')
    variables={int(r[0]):r for r in db.select('''SELECT v.variableid,v.nodatavalue,u.unitsabbreviation
        FROM leo_west.variables v JOIN leo_west.units u ON v.variableunitsid=u.unitsid
        WHERE v.variableid IN (9,56)''')}
    for c in channels:
        r=sensors[c['sensorid']];v=variables[c['variableid']]
        if r[1]!=c['sensorcode'] or str(r[2]).upper()!=c['table'].split('.')[1] or v[2]!=c['units']:
            raise ValueError(f"Oracle identity/units conflict: {c['sensorid']}")
        if float(r[6])!=c['x'] or float(r[7])!=c['y']:
            raise ValueError(f"Oracle horizontal coordinates conflict: {c['sensorid']}")
        c['oracle_location']=dict(zip(('localx','localy','localz','boxx','boxy','boxz','dlevel'),r[3:]))
        c['oracle_nodata']=str(v[1])
        c['geometry_note']='Excel air height .25m not established by Oracle BOXZ' if c['kind']=='LI-COR' else ''
        if c['kind']=='GMM222' and abs(float(r[8])+c['z'])>1e-8:
            raise ValueError(f"Oracle basalt depth conflict: {c['sensorid']}")
    schemas={}
    for table in sorted({c['table'] for c in channels}):
        short=table.split('.')[1]
        cols=db.select('''SELECT column_name,data_type,data_precision,data_scale,nullable
            FROM all_tab_columns WHERE owner='LEO_WEST' AND table_name=:t ORDER BY column_id''',{'t':short})
        if not {'VALUEID','LOCALDATETIME','DATAVALUE','SENSORID','VARIABLEID'} <= {r[0] for r in cols}:
            raise ValueError('Required raw fields missing')
        if any(not re.fullmatch('[A-Z][A-Z0-9_]*',r[0]) for r in cols):raise ValueError('Invalid column name')
        # Confirm the continuation key, not just a guessed field named VALUEID.
        keys=db.select('''SELECT cc.column_name FROM all_constraints c JOIN all_cons_columns cc
            ON c.owner=cc.owner AND c.constraint_name=cc.constraint_name
            WHERE c.owner='LEO_WEST' AND c.table_name=:t AND c.constraint_type='P'
            AND c.status='ENABLED' ORDER BY cc.position''',{'t':short})
        if keys!=[('VALUEID',)]:raise ValueError('Unique VALUEID continuation key not established')
        if any(r[4]!='N' for r in cols if r[0] in ('VALUEID','LOCALDATETIME')):
            raise ValueError('Nullable continuation key unsupported')
        schemas[table]=[dict(zip(('name','type','precision','scale','nullable'),r)) for r in cols]
    return schemas


def month_bounds(start,end):
    current=start
    while current<end:
        next_month=(current.replace(day=1,hour=0,minute=0,second=0,microsecond=0)+timedelta(days=32)).replace(day=1)
        following=min(next_month,end)
        yield current,following;current=following


def new_block(table,start,end,floor=None,parent=None):
    return dict(id=uuid.uuid4().hex[:16],table=table,start=start.isoformat(),end=end.isoformat(),
                floor=floor,parent=parent,status='pending',parts=[],errors=[])


def block_filter(block,channels,key=None):
    params={'start_time':datetime.fromisoformat(block['start']),'end_time':datetime.fromisoformat(block['end'])}
    clause=channel_where(channels)+' AND localdatetime>=:start_time AND localdatetime<:end_time'
    key=key or block.get('floor')
    if key:
        clause+=' AND (localdatetime>:kt OR (localdatetime=:kt AND valueid>:ki))'
        params.update(kt=datetime.fromisoformat(key[0]),ki=int(key[1]))
    return clause,params


def count_block(db,block,channels):
    clause,params=block_filter(block,channels)
    rows=db.select(f"SELECT sensorid,COUNT(*) FROM {block['table']} WHERE {clause} GROUP BY sensorid",params)
    return {str(int(sid)):int(n) for sid,n in rows}


def persisted_counts(block):
    out=Counter()
    for part in block['parts']:out.update(part['sensor_counts'])
    return dict(out)


def verify_parts(version,block):
    for part in block['parts']:
        path=version/part['file']
        if not path.exists() or digest(path)!=part['sha256'] or pq.ParquetFile(path).metadata.num_rows!=part['rows']:
            raise ValueError('Confirmed raw part damaged: '+str(path))


def fetch_block(db,manifest,block,version,save):
    """Checkpoint each complete fetchmany page; restart uses its strict last key."""
    channels=[c for c in manifest['channels'] if c['table']==block['table']]
    verify_parts(version,block)
    before=count_block(db,block,channels)
    block['expected_counts']=before;save()
    if (sum(before.values())>MAX_BLOCK_ROWS and not block['parts']
            and params_span(block)>timedelta(days=1)):
        return 'split'
    key=block['parts'][-1]['last_key'] if block['parts'] else block.get('floor')
    clause,params=block_filter(block,channels,key)
    columns=[c['name'] for c in manifest['schemas'][block['table']]]
    sql='SELECT '+','.join('"'+c+'"' for c in columns)+f" FROM {block['table']} WHERE {clause} ORDER BY localdatetime,valueid"
    pos={name:columns.index(name) for name in columns}
    previous=(datetime.fromisoformat(key[0]),int(key[1])) if key else None
    block['status']='running';save()
    for rows in db.stream(sql,params):
        counts=Counter();first_key=None
        for row in rows:
            current=(row[pos['LOCALDATETIME']],int(row[pos['VALUEID']]))
            if previous is not None and current<=previous:raise ValueError('Non-increasing continuation key')
            if not params['start_time']<=current[0]<params['end_time']:raise ValueError('Time boundary leak')
            sid=int(row[pos['SENSORID']])
            if sid not in {c['sensorid'] for c in channels}:raise ValueError('Channel boundary leak')
            if int(row[pos['VARIABLEID']])!=channels[0]['variableid']:raise ValueError('Variable mismatch')
            first_key=first_key or current;previous=current;counts[str(sid)]+=1
        arrays={name.lower():[row[i] for row in rows] for i,name in enumerate(columns)}
        # Individual parts can have different Decimal scales; never cast raw values to float.
        table=pa.Table.from_pydict(arrays)
        relative=Path('raw')/block['id']/f"part-{len(block['parts']):06d}.parquet"
        path=version/relative;path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_suffix('.parquet.tmp');pq.write_table(table,temp,compression='zstd')
        with temp.open('rb') as stream:os.fsync(stream.fileno())
        if pq.ParquetFile(temp).metadata.num_rows!=len(rows):raise ValueError('Parquet row count mismatch')
        os.replace(temp,path);sync_dir(path.parent)
        block['parts'].append(dict(file=str(relative),rows=len(rows),sha256=digest(path),
            sensor_counts=dict(counts),first_key=[first_key[0].isoformat(),str(first_key[1])],
            last_key=[previous[0].isoformat(),str(previous[1])]))
        save()
        LOG.info('saved %s %s rows=%d block=%s parts=%d',block['table'],block['start'],len(rows),block['id'],len(block['parts']))
    after=count_block(db,block,channels)
    if before!=after or persisted_counts(block)!=after:
        raise ValueError('Oracle before/after counts or saved counts differ; parts preserved')
    block.update(status='complete',completed_at=now(),verified_counts=persisted_counts(block));save()
    LOG.info('complete %s %s rows=%d',block['table'],block['start'],sum(after.values()))
    return 'complete'


def params_span(block):
    return datetime.fromisoformat(block['end'])-datetime.fromisoformat(block['start'])


def split_block(manifest,block):
    """Split only the remaining key range; never discard confirmed pages."""
    floor=block['parts'][-1]['last_key'] if block['parts'] else block.get('floor')
    start=datetime.fromisoformat(floor[0] if floor else block['start']);end=datetime.fromisoformat(block['end'])
    span=end-start
    if span<=timedelta(days=1):return False
    step=timedelta(days=7) if span>timedelta(days=7) else timedelta(days=1)
    children=[];cursor=start
    while cursor<end:
        nxt=min(cursor+step,end)
        child=new_block(block['table'],cursor,nxt,floor if cursor==start else None,block['id'])
        children.append(child);cursor=nxt
    manifest['blocks'].extend(children)
    block.update(status='split',children=[c['id'] for c in children]);return True


def resolve_parents(db,manifest,save):
    byid={b['id']:b for b in manifest['blocks']}
    for block in reversed(manifest['blocks']):
        if block['status']!='split':continue
        children=[byid[i] for i in block['children']]
        if not all(c['status']=='complete' for c in children):continue
        counts=Counter(persisted_counts(block))
        for child in children:counts.update(child['verified_counts'])
        selected=[c for c in manifest['channels'] if c['table']==block['table']]
        try:
            current=count_block(db,block,selected)
            if dict(counts)!=current:raise ValueError('Split block total no longer matches Oracle')
            block.update(status='complete',verified_counts=dict(counts));save()
        except Exception as exc:
            block['errors'].append(dict(at=now(),type=type(exc).__name__,message=str(exc)))
            db.close();save()


def initialize(db,manifest,save):
    if 'schemas' not in manifest:
        manifest['schemas']=inspect_metadata(db,manifest['channels']);save()
    for table in sorted(manifest['schemas']):
        if table in manifest.setdefault('bounds',{}):continue
        selected=[c for c in manifest['channels'] if c['table']==table]
        # No imposed year bound and no data-value filter.
        rows=db.select(f'''SELECT sensorid,COUNT(*),MIN(localdatetime),MAX(localdatetime)
            FROM {table} WHERE {channel_where(selected)} GROUP BY sensorid''')
        bounds={str(int(sid)):dict(rows=int(n),first=start.isoformat(),last=end.isoformat()) for sid,n,start,end in rows}
        manifest['bounds'][table]=bounds
        if rows:
            start=min(r[2] for r in rows);end=max(r[3] for r in rows)+timedelta(seconds=1)
            manifest['blocks'].extend(new_block(table,a,b) for a,b in month_bounds(start,end))
        save();LOG.info('Oracle bounds %s channels=%d rows=%d',table,len(bounds),sum(b['rows'] for b in bounds.values()))
    manifest['initialized']=True;save()


def download(db,manifest,version,save):
    # This pass also revisits failed blocks from a previous invocation.
    for block in manifest['blocks']:
        if block['status'] in ('complete','split'):continue
        for attempt in range(RETRIES):
            try:
                result=fetch_block(db,manifest,block,version,save)
                if result=='split':
                    if split_block(manifest,block):save();break
                    # A very dense single day still streams with key checkpoints.
                    raise ValueError('Dense one-day block requires streaming')
                block['verified_counts']=persisted_counts(block);save();break
            except Exception as exc:
                LOG.exception('block %s attempt %d/%d',block['id'],attempt+1,RETRIES)
                block['errors'].append(dict(at=now(),type=type(exc).__name__,message=str(exc)))
                block['status']='failed';save();db.close()
                if attempt+1<RETRIES:time.sleep(min(2**attempt,10))
        else:
            if split_block(manifest,block):save()
        # Appended children are visited by this same list iterator.
    resolve_parents(db,manifest,save)


def status(archive):
    pointer=archive/'latest.json'
    if not pointer.exists():return {'status':'no archive yet'}
    version=archive/json.loads(pointer.read_text())['version']
    manifest=json.loads((version/'manifest.json').read_text())
    blocks=manifest['blocks'];done=Counter();saved=Counter()
    for block in blocks:
        saved.update(persisted_counts(block))
    for c in manifest['channels']:
        roots=[b for b in blocks if not b.get('parent') and b['table']==c['table']]
        bounds=manifest.get('bounds',{}).get(c['table'],{})
        if c['table'] in manifest.get('bounds',{}) and (str(c['sensorid']) not in bounds or all(b['status']=='complete' for b in roots)):
            done[c['sensorid']]=1
    return dict(version=str(version),status=manifest['status'],updated_at=manifest['updated_at'],
                channels_total=len(manifest['channels']),channels_complete=len(done),rows_saved=sum(saved.values()),
                channels_with_saved_rows=len(saved),saved_rows_by_sensorid=dict(saved),
                blocks_by_status=dict(Counter(b['status'] for b in blocks)),
                error_events=sum(len(b['errors']) for b in blocks)+len(manifest.get('errors',[])),
                recent_errors=(manifest.get('errors',[])+[
                    dict(block=b['id'],start=b['start'],**b['errors'][-1])
                    for b in blocks if b['errors']])[-5:],log=str(version/'download.log'),
                diagnostic_status=manifest.get('diagnostics','not started'))


def refresh(archive,skip_diagnostics=False):
    archive.mkdir(parents=True,exist_ok=True)
    with (archive/'writer.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('A co2-refresh process already owns this archive')
        pointer=archive/'latest.json'
        if pointer.exists():
            version=archive/json.loads(pointer.read_text())['version']
            manifest=json.loads((version/'manifest.json').read_text())
        else:manifest=None
        if manifest is None or manifest['status']=='complete':
            name=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6]
            version=archive/name;version.mkdir()
            manifest=dict(format_version=1,status='initializing',created_at=now(),updated_at=now(),
                channels=workbook_channels(),workbook_sha256=digest(WORKBOOK),blocks=[],bounds={},
                source_consistency='Per-block before/after counts; live Oracle, not one global SCN snapshot',
                timezone='Oracle localdatetime; timezone and clock correspondence unconfirmed',
                numeric_storage='Oracle NUMBER/FLOAT fetched as Decimal; independent Parquet part schemas',
                continuation_key=['LOCALDATETIME','VALUEID'])
            atomic_json(version/'manifest.json',manifest);atomic_json(pointer,{'version':name})
        handler=logging.FileHandler(version/'download.log');handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        LOG.addHandler(handler);LOG.setLevel(logging.INFO)
        def save():
            manifest['updated_at']=now();atomic_json(version/'manifest.json',manifest)
        db=Oracle()
        try:
            if not manifest.get('initialized'):
                for attempt in range(RETRIES):
                    try:initialize(db,manifest,save);break
                    except Exception:
                        LOG.exception('metadata attempt %d/%d',attempt+1,RETRIES);db.close()
                        if attempt+1==RETRIES:raise
                        time.sleep(2)
            manifest['status']='downloading';save()
            download(db,manifest,version,save)
            complete=all(b['status']=='complete' for b in manifest['blocks'])
            if not complete:
                manifest['status']='incomplete';save();LOG.error('Incomplete blocks retained; run make co2-refresh to retry');return 2
            for block in manifest['blocks']:verify_parts(version,block)
            manifest['download_complete_at']=now();manifest['status']='download_complete';save()
            db.close()
            # Run the diagnostic process only when all 53 source channels are complete.
            if not skip_diagnostics:
                manifest['diagnostics']='running';save()
                command=[sys.executable,'-B',str(Path(__file__).with_name('co2_full_diagnostics.py')),str(version)]
                result=subprocess.run(command,stdout=handler.stream,stderr=handler.stream)
                if result.returncode:
                    manifest['diagnostics']='failed';save();return result.returncode
                manifest['diagnostics']='complete'
            manifest['status']='complete';manifest['completed_at']=now();save()
            LOG.info('FULL REFRESH COMPLETE %s',json.dumps(status(archive)))
            print(json.dumps(status(archive),indent=2));return 0
        except Exception as exc:
            manifest.setdefault('errors',[]).append(dict(at=now(),type=type(exc).__name__,message=str(exc)))
            manifest['status']='incomplete';save();LOG.exception('Stopped safely; confirmed progress retained');return 2
        finally:db.close();handler.close();LOG.removeHandler(handler)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['refresh','status'])
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--skip-diagnostics',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.action=='status':print(json.dumps(status(args.root),ensure_ascii=False,indent=2));return 0
    return refresh(args.root,args.skip_diagnostics)


if __name__=='__main__':sys.exit(main())
