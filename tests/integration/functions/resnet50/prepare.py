"""Pack-time hook: weights into $TORCH_HOME, sample image into the package (cwd)."""
import urllib.request
from torchvision.models import resnet50

resnet50(pretrained=True)  # downloads into $TORCH_HOME/hub/checkpoints
urllib.request.urlretrieve("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
