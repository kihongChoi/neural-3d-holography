import base64
from functools import partial
import hashlib
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

from benchtop.live import Handler
from benchtop.paper_view import RUN


class CompletedPaperViewerTests(unittest.TestCase):
    def test_completed_result_replaces_empty_prepare_stream_and_checks_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root/'results'
            run = results/RUN
            live = run/'live'/'prepare'
            live.mkdir(parents=True)
            (live.parent/'current.json').write_text(json.dumps({'run_id': 'prepare'}))
            (live/'state.json').write_text(json.dumps(dict(run_id='prepare', status='completed',
                                                         updated_at=10, ended_at=10, streams={})))
            (live/'heartbeat.json').write_text('{"time":10}')
            (run/'evaluation').mkdir()
            files = {'completed_all.json': {'status': 'completed', 'updated_at': 200},
                     'evaluation/summary.json': {'metric': .123}, 'training.jsonl': {'epoch': 61}}
            sources = []
            for name, value in files.items():
                path = run/name
                path.write_text(json.dumps(value))
                sources.append(dict(path=path.relative_to(root).as_posix(),
                                    sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            original = {str(p): p.read_bytes() for p in run.rglob('*') if p.is_file()}
            docs = root/'docs/learning-20261024'
            docs.mkdir(parents=True)
            preview = b'RIFF-webp-test'
            data = dict(sources=sources, means={'metric': .123}, targets=[dict(id='0822', means={},
                        images={'target': 'data:image/webp;base64,'+base64.b64encode(preview).decode()})])
            (docs/'paper-results-data.json').write_text(json.dumps(data))
            server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Handler, root=results))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            prefix = f'http://127.0.0.1:{server.server_port}'
            def read(path):
                with urlopen(prefix+path, timeout=5) as response:
                    return response.read()
            try:
                state = json.loads(read('/api/state?run='+RUN))
                self.assertEqual(state['ended_at'], 200)
                self.assertIn('/paper/', state['result_viewer'])
                result = json.loads(read('/api/paper?run='+RUN))
                self.assertEqual(result['means']['metric'], .123)
                self.assertNotIn('images', result['targets'][0])
                self.assertEqual(read('/api/paper-image?run='+RUN+'&target=0822&image=target'), preview)
                for suffix in ['&target=../../private&image=target', '&target=0822&image=../../private']:
                    with self.assertRaises(HTTPError) as error:
                        read('/api/paper-image?run='+RUN+suffix)
                    self.assertEqual(error.exception.code, 404)
                self.assertEqual(original, {str(p): p.read_bytes() for p in run.rglob('*') if p.is_file()})
                (run/'training.jsonl').write_text('{"epoch":62,"changed":true}')
                with self.assertRaises(HTTPError) as error:
                    read('/api/paper?run='+RUN)
                self.assertEqual(error.exception.code, 404)
                self.assertIn('changed', error.exception.read().decode())
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == '__main__':
    unittest.main()
