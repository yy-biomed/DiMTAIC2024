import torch
from torchvision.io import read_image
from torchvision.transforms import v2
import torch.nn.functional as F
from pathlib import Path
import pandas as pd

from model import get_resnet50


def infer(model, img_dir, img_shape, device):
    model.eval()

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
        img = img.to(device)

        with torch.no_grad():
            pred = model(img)
            pred = F.sigmoid(pred)
            pred = pred.item()
        case.append(basename)
        prob.append(pred)

    return pd.DataFrame({'case': case, 'prob': prob})


if __name__ == '__main__':

    ############################################################################
    # test_dir = '/tcdata/test/img/'
    test_dir = '../../tcdata/test/img/'  # 测试路径
    # submit_dir = '/app/submit/'
    submit_dir = '../../submit/'  # 测试路径
    log_dir = 'log'

    # n_class = 1
    n_class = 3  # 测试时使用3
    ############################################################################


    Path(submit_dir).mkdir(parents=True, exist_ok=True)

    using_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    classifi_model = get_resnet50(n_class)
    classifi_model.load_state_dict(torch.load('%s/best.pth' % log_dir, weights_only=True))
    classifi_model.to(using_device)

    infer(classifi_model, test_dir, (512, 512), using_device).to_csv('label.csv', index=False)
