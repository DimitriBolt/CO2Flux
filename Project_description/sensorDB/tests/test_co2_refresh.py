from datetime import datetime,timedelta
from decimal import Decimal
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import pyarrow.parquet as pq
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import co2_refresh as r

COLUMNS=['VALUEID','DATAVALUE','LOCALDATETIME','SENSORID','VARIABLEID']
CHANNEL=dict(table='LEO_WEST.DATAVALUESLICOR',sensorid=1275,variableid=56)
START=datetime(2025,1,1)


class Fake:
    def __init__(self,rows,fail=False):self.rows=rows;self.fail=fail;self.queries=[]
    def select(self,sql,params):
        rows=self.selected(params)
        return [(Decimal(1275),Decimal(len(rows)))] if rows else []
    def selected(self,params):
        return [row for row in self.rows if params['start_time']<=row[2]<params['end_time'] and
                ('kt' not in params or (row[2],int(row[0]))>(params['kt'],params['ki']))]
    def stream(self,sql,params):
        self.queries.append(params)
        rows=self.selected(params)
        for i in range(0,len(rows),2):
            yield rows[i:i+2]
            if self.fail:self.fail=False;raise ConnectionError('injected disconnect')
    def close(self):pass


def fixture():
    # Repeated clocks and identical observations have distinct real primary keys.
    values=[Decimal('-9999'),Decimal('400.12345'),Decimal('400.12345'),None,Decimal('9000')]
    return [(Decimal(i+1),value,START+timedelta(seconds=i//2),Decimal(1275),Decimal(56)) for i,value in enumerate(values)]


class RefreshTests(unittest.TestCase):
    def manifest(self):
        block=r.new_block(CHANNEL['table'],START,START+timedelta(days=1))
        return dict(channels=[CHANNEL],schemas={CHANNEL['table']:[{'name':c} for c in COLUMNS]},blocks=[block]),block

    def test_small_block_preserves_all_values_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            version=Path(directory);m,b=self.manifest();save=lambda:r.atomic_json(version/'manifest.json',m)
            self.assertEqual(r.fetch_block(Fake(fixture()),m,b,version,save),'complete')
            loaded=[]
            for part in b['parts']:loaded.extend(pq.read_table(version/part['file']).to_pylist())
            self.assertEqual([x['datavalue'] for x in loaded],[x[1] for x in fixture()])
            self.assertEqual(len(loaded),5)
            self.assertEqual(len({x['valueid'] for x in loaded}),5)
            self.assertEqual(b['verified_counts'],{'1275':5})
            r.verify_parts(version,b)

    def test_disconnect_then_reload_manifest_keeps_completed_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            version=Path(directory);m,b=self.manifest();save=lambda:r.atomic_json(version/'manifest.json',m)
            db=Fake(fixture(),fail=True)
            with self.assertRaises(ConnectionError):r.fetch_block(db,m,b,version,save)
            first=b['parts'][0].copy();self.assertEqual(first['rows'],2)
            # Simulate a new process: no in-memory cursor/state survives.
            resumed=json.loads((version/'manifest.json').read_text());block=resumed['blocks'][0]
            db=Fake(fixture());save2=lambda:r.atomic_json(version/'manifest.json',resumed)
            r.fetch_block(db,resumed,block,version,save2)
            self.assertEqual(block['parts'][0],first)
            self.assertEqual(r.digest(version/first['file']),first['sha256'])
            self.assertEqual(db.queries[0]['ki'],2)
            self.assertEqual(sum(p['rows'] for p in block['parts']),5)

    def test_completed_other_block_survives_failure_and_run(self):
        with tempfile.TemporaryDirectory() as directory:
            version=Path(directory);m,b=self.manifest();save=lambda:r.atomic_json(version/'manifest.json',m)
            r.fetch_block(Fake(fixture()),m,b,version,save);old=json.dumps(b,default=str)
            second=r.new_block(CHANNEL['table'],START+timedelta(days=1),START+timedelta(days=2));m['blocks'].append(second)
            r.download(Fake(fixture()),m,version,save)
            self.assertEqual(json.dumps(b,default=str),old)
            self.assertEqual(second['status'],'complete')

    def test_partial_split_uses_strict_floor_and_half_open_boundaries(self):
        m,b=self.manifest();b['end']=(START+timedelta(days=16)).isoformat()
        b['parts']=[dict(last_key=[(START+timedelta(days=2)).isoformat(),'100'])]
        self.assertTrue(r.split_block(m,b))
        children=m['blocks'][1:]
        self.assertEqual(children[0]['floor'],b['parts'][0]['last_key'])
        self.assertEqual(children[-1]['end'],b['end'])
        for left,right in zip(children,children[1:]):self.assertEqual(left['end'],right['start'])
        self.assertTrue(all(c['floor'] is None for c in children[1:]))

    def test_crash_before_manifest_commit_orphan_not_counted_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            version=Path(directory);m,b=self.manifest()
            save=lambda:r.atomic_json(version/'manifest.json',m)
            save()
            original=save
            def crash():
                if b['parts']:raise RuntimeError('crash after durable file before durable manifest')
                original()
            with self.assertRaises(RuntimeError):r.fetch_block(Fake(fixture()),m,b,version,crash)
            resumed=json.loads((version/'manifest.json').read_text());block=resumed['blocks'][0]
            r.fetch_block(Fake(fixture()),resumed,block,version,lambda:r.atomic_json(version/'manifest.json',resumed))
            self.assertEqual(sum(p['rows'] for p in block['parts']),5)
            self.assertEqual(len(list((version/'raw').rglob('*.parquet'))),3)

    def test_corrupt_completed_page_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            version=Path(directory);m,b=self.manifest()
            r.fetch_block(Fake(fixture()),m,b,version,lambda:None)
            (version/b['parts'][0]['file']).write_bytes(b'bad')
            with self.assertRaises(ValueError):r.verify_parts(version,b)


if __name__=='__main__':unittest.main()

class LifecycleTests(unittest.TestCase):
    def test_refresh_resumes_incomplete_then_creates_new_complete_version(self):
        def initialize(db,manifest,save):
            manifest['schemas']={CHANNEL['table']:[{'name':c} for c in COLUMNS]}
            manifest['bounds']={CHANNEL['table']:{'1275':{'rows':5}}}
            manifest['blocks']=[r.new_block(CHANNEL['table'],START,START+timedelta(days=1))]
            manifest['initialized']=True;save()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'archive';book=Path(directory)/'book';book.write_text('fixture')
            db=Fake(fixture(),fail=True)
            with patch.object(r,'WORKBOOK',book),patch.object(r,'workbook_channels',return_value=[CHANNEL]), \
                 patch.object(r,'Oracle',return_value=db),patch.object(r,'initialize',side_effect=initialize), \
                 patch.object(r,'RETRIES',1):
                self.assertEqual(r.refresh(root,True),2)
                first=json.loads((root/'latest.json').read_text())['version']
                self.assertEqual(r.status(root)['rows_saved'],2)
                self.assertEqual(r.refresh(root,True),0)
                self.assertEqual(json.loads((root/'latest.json').read_text())['version'],first)
                self.assertEqual(r.status(root)['rows_saved'],5)
                self.assertEqual(r.refresh(root,True),0)
                second=json.loads((root/'latest.json').read_text())['version']
                self.assertNotEqual(first,second)
                self.assertTrue((root/first/'manifest.json').exists())
                self.assertEqual(r.status(root)['rows_saved'],5)
