from pathlib import Path
from datetime import datetime,timedelta
import sys,tempfile,unittest
from unittest.mock import patch
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import co2_refresh as r
import co2_full_diagnostics as d

class FullDiagnosisTests(unittest.TestCase):
    def test_all_16_verticals_and_raw_preservation(self):
        channels=r.workbook_channels()
        with tempfile.TemporaryDirectory() as directory:
            version=Path(directory);start=datetime(2025,1,1);end=start+timedelta(days=1)
            blocks=[]
            for table in sorted({c['table'] for c in channels}):
                rows=[]
                for c in channels:
                    for k in range(24):
                        rows.append(dict(valueid=len(rows)+1,datavalue=400.+k%5,sensorid=c['sensorid'],
                            variableid=c['variableid'],localdatetime=start+timedelta(hours=k)))
                    if c['table']!=table:del rows[-24:]
                path=version/(table.split('.')[1]+'.parquet');pq.write_table(pa.Table.from_pylist(rows),path)
                blocks.append(dict(status='complete',table=table,start=start.isoformat(),end=end.isoformat(),
                    id=table.split('.')[1],parts=[dict(file=path.name,sha256=r.digest(path),rows=len(rows))]))
            manifest=dict(channels=channels,blocks=blocks,download_complete_at='fixture')
            r.atomic_json(version/'manifest.json',manifest)
            original={p:r.digest(p) for p in version.glob('*.parquet')}
            with patch.object(d,'update_document') as document:
                d.diagnose(version)
                document.assert_called_once()
            candidates=pd.read_csv(version/'diagnostics/vertical_candidates.csv')
            self.assertEqual(len(candidates),16)
            self.assertTrue(candidates.common_retained_days.eq(1).all())
            self.assertTrue(candidates.common_without_review_days.eq(0).all())
            self.assertEqual(len(pd.read_csv(version/'diagnostics/channels.csv')),53)
            self.assertEqual(len(pd.read_csv(version/'diagnostics/air_distances.csv')),80)
            self.assertEqual({p:r.digest(p) for p in original},original)
            self.assertEqual(set(candidates.depths_cm),{'5,20,35','5,20,50'})

    def test_seasons_and_real_daily_runs_do_not_bridge_gaps(self):
        dates=pd.to_datetime(['2024-12-31','2025-01-01','2025-01-03'])
        frame=pd.DataFrame({'raw_rows':[1,2,1]},index=dates)
        self.assertEqual(list(d.periods(frame)['season']),['2025-DJF']*3)
        runs=d.runs(dates)
        self.assertEqual(runs.days.tolist(),[2,1])
        self.assertEqual(runs.end_exclusive.iloc[0],pd.Timestamp('2025-01-02'))
