import os
import torch
from multiprocessing import cpu_count
from lib.infer.modules.vc.modules import VC
from lib.infer.infer_libs.csvutil import CSVutil

models_dir = os.path.join(os.getcwd(), "models")
os.environ["index_root"] = "logs"
os.environ["rmvpe_root"] = "assets/rmvpe"
os.environ["fcpe_root"] = "assets/fcpe"

class Configs:
    def __init__(self, device, is_half):
        self.device = device
        self.is_half = is_half
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                    ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                    or "P40" in self.gpu_name.upper()
                    or "1060" in self.gpu_name
                    or "1070" in self.gpu_name
                    or "1080" in self.gpu_name
            ):
                print("16 series/10 series P40 forced single precision")
                self.is_half = False
                for config_file in ["32k.json", "40k.json", "48k.json"]:
                    with open(BASE_DIR / "src" / "configs" / config_file, "r") as f:
                        strr = f.read().replace("true", "false")
                    with open(BASE_DIR / "src" / "configs" / config_file, "w") as f:
                        f.write(strr)
                with open(BASE_DIR / "src" / "trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open(BASE_DIR / "src" / "trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)
            else:
                self.gpu_name = None
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
            if self.gpu_mem <= 4:
                with open(BASE_DIR / "src" / "trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open(BASE_DIR / "src" / "trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)
        elif torch.backends.mps.is_available():
            print("No supported N-card found, use MPS for inference")
            self.device = "mps"
        else:
            print("No supported N-card found, use CPU for inference")
            self.device = "cpu"
            self.is_half = True

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            # 6G memory config
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5G memory config
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem != None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        return x_pad, x_query, x_center, x_max

configs = Configs('cuda:0', True)
vc = VC(configs)

def get_model(voice_model):
    model_dir = os.path.join(models_dir, voice_model)
    model_filename, index_filename = None, None
    for file in os.listdir(model_dir):
        ext = os.path.splitext(file)[1]
        if ext == '.pth':
            model_filename = file
        if ext == '.index':
            index_filename = file

    if model_filename is None:
        print(f'No model file exists in {models_dir}.')

    return os.path.join(model_dir, model_filename), os.path.join(model_dir, index_filename) if index_filename else ''

def infer_audio(MODEL_NAME, SOUND_PATH, F0_CHANGE, F0_METHOD, CREPE_HOP_LENGTH, INDEX_RATE, FILTER_RADIUS, RMS_MIX_RATE, PROTECT, AUDIO_FORMAT, FORMANT_SHIFT, QUEFRENCY, TIMBRE, F0_AUTOTUNE, MIN_PITCH, MAX_PITCH):
    PTH_PATH, INDEX_PATH = get_model(MODEL_NAME)
    
    if not FORMANT_SHIFT:
        DoFormant = False
        Quefrency = 0.0
        Timbre = 0.0
        CSVutil(
            "lib/csvdb/formanting.csv", "w+", "formanting", DoFormant, Quefrency, Timbre
        )
    else:
        DoFormant = True
        Quefrency = QUEFRENCY
        Timbre = TIMBRE
        CSVutil(
            "lib/csvdb/formanting.csv", "w+", "formanting", DoFormant, Quefrency, Timbre
        )

    minpitch_txtbox = MIN_PITCH
    maxpitch_txtbox = MAX_PITCH

    vc_data = vc.get_vc(PTH_PATH, PROTECT, 0.5)

    conversion_data = vc.vc_single(
        0,
        SOUND_PATH,
        F0_CHANGE,
        None,
        F0_METHOD,
        INDEX_PATH,
        INDEX_PATH,
        INDEX_RATE,
        FILTER_RADIUS,
        0,
        RMS_MIX_RATE,
        PROTECT,
        AUDIO_FORMAT,
        False,
        CREPE_HOP_LENGTH,
        MIN_PITCH,
        minpitch_txtbox,
        MAX_PITCH,
        maxpitch_txtbox,
        F0_AUTOTUNE,
    )

    return conversion_data