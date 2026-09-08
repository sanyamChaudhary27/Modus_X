import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import enwik8_data as data


class AcquisitionTests(unittest.TestCase):
    def test_download_cache_and_integrity(self):
        payload=b'byte fixture\x00\xff\r\n'
        archive=io.BytesIO()
        with zipfile.ZipFile(archive,'w') as z: z.writestr('nested/enwik8',payload)
        with tempfile.TemporaryDirectory() as tmp, patch.object(data,'SIZE',len(payload)), patch.object(data,'SHA256',hashlib.sha256(payload).hexdigest()):
            with patch.object(data.urllib.request,'urlopen',return_value=io.BytesIO(archive.getvalue())) as network:
                result=data.ensure(roots=(),work=tmp)
                self.assertEqual(result.read_bytes(),payload)
                self.assertEqual(data.ensure(roots=(),work=tmp),result)
                self.assertEqual(network.call_count,1)
            bad=Path(tmp)/'bad.zip'
            with zipfile.ZipFile(bad,'w') as z: z.writestr('enwik8',b'x'*len(payload))
            with self.assertRaisesRegex(ValueError,'SHA-256'): data.extract(bad,result)
            self.assertEqual(result.read_bytes(),payload)
            self.assertFalse(result.with_suffix('.extracting').exists())


if __name__=='__main__': unittest.main()
