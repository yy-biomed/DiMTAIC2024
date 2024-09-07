import torch
from torchvision.io import read_image
from torchvision.transforms import v2
import torch.nn.functional as F
from pathlib import Path
import pandas as pd

from model import get_resnet50


def infer(model, img_dir, img_shape, device):
    transforms = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Grayscale(num_output_channels=3),
        v2.CenterCrop(img_shape),
    ])

    case = []
    prob = []
    for f in Path(img_dir).glob('*.png'):
        basename = f.stem
        img = read_image(str(f))
        img = transforms(img)
        img = img.reshape(1, *img.shape)
        img = img.to(device)

        with torch.no_grad():
            pred = model(img)
            pred = F.softmax(pred, dim=1)
            pred = pred[0, 1].item()
        case.append(basename)
        prob.append(pred)

    return pd.DataFrame({'case': case, 'prob': prob})


if __name__ == '__main__':

    ############################################################################
    test_dir = '/tcdata/test/img/'
    submit_dir = '/app/submit/'
    log_dir = '/app/ResNet/log'

    n_class = 2
    ############################################################################

    Path(submit_dir).mkdir(parents=True, exist_ok=True)

    using_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    classifi_model = get_resnet50(n_class)
    classifi_model.load_state_dict(torch.load('%s/best.pth' % log_dir, weights_only=True))
    classifi_model.to(using_device)

    classifi_model.eval()
    infer(classifi_model, test_dir, (512, 512), using_device).to_csv('%s/label.csv' % submit_dir, index=False)

    print('Done!')
