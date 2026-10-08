"""Pack-time hook: weights into $TORCH_HOME, sample image into the package (cwd)."""
import hashlib
import urllib.request
from torchvision.models import resnet50

# pinned pytorch/hub commit and checksum
URL = "https://github.com/pytorch/hub/raw/c3beaae7d32fca2a23fec30aa7938ef5c9b6e5d5/images/dog.jpg"
SHA256 = "f3f87bb8ab3c26c7ecfd3ac60421d7f32b0503d1d6c5baf8bac42ed93d86351a"

resnet50(pretrained=True)  # downloads into $TORCH_HOME/hub/checkpoints
data = urllib.request.urlopen(URL).read()
assert hashlib.sha256(data).hexdigest() == SHA256, "dog.jpg checksum mismatch"
open("dog.jpg", "wb").write(data)
