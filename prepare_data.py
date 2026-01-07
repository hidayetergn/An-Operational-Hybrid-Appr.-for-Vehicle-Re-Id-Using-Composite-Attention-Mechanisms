import os
import zipfile
import yaml
import sys


def load_config():
    # 1. Kodun çalıştığı klasörü bul (Absolute Path)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config.yaml")

    print(f"[DEBUG] Looking for config file at: {config_path}")

    # 2. Dosya var mı kontrolü
    if not os.path.exists(config_path):
        print(f"[ERROR] Configuration file NOT FOUND at: {config_path}")
        print("Please make sure 'config.yaml' is in the same folder as 'prepare_data.py'.")
        sys.exit(1)

    # 3. Dosya içeriği okuma
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 4. Dosya boş mu kontrolü
    if cfg is None:
        print(f"[ERROR] 'config.yaml' exists but is EMPTY or INVALID.")
        print("Please paste the configuration content into the file and save it (Ctrl+S).")
        sys.exit(1)

    return cfg


def extract_zip(zip_path, extract_to):
    # Zip yolunu kontrol et
    if not os.path.exists(zip_path):
        print(f"[ERROR] Zip file not found at: {zip_path}")
        print("Check 'INPUT_PATHS: RAW_ZIP_FILE' in config.yaml")
        # Hata olsa bile kullanıcı görsün diye exit yapmıyorum, uyarı veriyorum
        return

    print(f"[INFO] Extracting dataset...")
    print(f"       Source: {zip_path}")
    print(f"       Dest:   {extract_to}")

    os.makedirs(extract_to, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"[INFO] Extraction successful.")
    except Exception as e:
        print(f"[ERROR] Zip extraction failed: {e}")
        sys.exit(1)


def process_list(input_txt, output_txt):
    if not os.path.exists(input_txt):
        print(f"[ERROR] List file not found at: {input_txt}")
        print("Check 'INPUT_PATHS: RAW_LIST_FILE' in config.yaml")
        return

    print(f"[INFO] Processing list file...")
    output_dir = os.path.dirname(output_txt)
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    with open(input_txt, "r", encoding="utf-8") as f_in, \
            open(output_txt, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line: continue

            filename = os.path.basename(line)
            name_no_ext = os.path.splitext(filename)[0]
            try:
                vid = int(name_no_ext.split("_")[0])
                f_out.write(f"{name_no_ext} {vid}\n")
                count += 1
            except:
                continue

    print(f"[INFO] List processed. {count} valid entries saved to {output_txt}")


if __name__ == "__main__":
    # Config Yükle
    cfg = load_config()

    print("[INFO] Config loaded successfully.")

    # Veriyi Çıkar
    extract_zip(
        zip_path=cfg['INPUT_PATHS']['RAW_ZIP_FILE'],
        extract_to=cfg['DATASET']['IMAGE_DIR']
    )

    # Listeyi Düzenle
    process_list(
        input_txt=cfg['INPUT_PATHS']['RAW_LIST_FILE'],
        output_txt=cfg['DATASET']['PROCESSED_LIST']
    )

    print("\n[SUCCESS] Data preparation complete.")