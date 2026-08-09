import concurrent.futures
import urllib.request
import os
import socket
from tqdm import tqdm
import PIL
from PIL import Image
import pandas as pd

socket.setdefaulttimeout(30)

DATA_DIR = "/home/users/ymathansingh/Master_Thesis/EUFCC-340K/data"

BASE_SAVE_ROOT = (
    "/home/users/ymathansingh/Master_Thesis/"
    "01_download_images/png/downloaded_images_png"
)

NUM_THREADS = 24


def download_image(url, img_name):
    try:
        urllib.request.urlretrieve(url, img_name)

    except Exception as e:
        print(f"Error downloading {img_name}: {e}")


def download_images(df, split, num_threads=24):
    """
    Download images for one split.
    Existing images are skipped.
    Downloads start immediately.
    """

    save_root = os.path.join(BASE_SAVE_ROOT, split)
    os.makedirs(save_root, exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_threads
    ) as executor:

        futures = []

        with tqdm(
            total=len(df),
            desc=f"downloading {split}"
        ) as progress:

            for _, row in df.iterrows():

                filename = row["idInSource"]
                url = row["#portraitMedia.original"]

                img_name = os.path.join(
                    save_root,
                    f"{filename}.png"
                )

                # Skip already downloaded images
                if os.path.exists(img_name):
                    progress.update(1)
                    continue

                future = executor.submit(
                    download_image,
                    url,
                    img_name
                )

                futures.append(future)

                # Avoid keeping too many pending tasks
                if len(futures) >= num_threads * 10:

                    for future in concurrent.futures.as_completed(futures):

                        try:
                            future.result()

                        except Exception as e:
                            print(f"Exception: {e}")

                        progress.update(1)

                    futures = []

            # Finish remaining downloads
            for future in concurrent.futures.as_completed(futures):

                try:
                    future.result()

                except Exception as e:
                    print(f"Exception: {e}")

                progress.update(1)


def validate_images(df, split):
    """
    Check downloaded images are present and valid.
    """

    save_root = os.path.join(
        BASE_SAVE_ROOT,
        split
    )

    print(f"Validating {split} images...")

    missing = []

    for _, row in tqdm(
        df.iterrows(),
        total=df.shape[0]
    ):

        filename = row["idInSource"]

        img_name = os.path.join(
            save_root,
            f"{filename}.png"
        )

        try:
            Image.open(img_name).close()

        except (
            FileNotFoundError,
            PIL.UnidentifiedImageError
        ):
            missing.append(filename)


    print(
        f"{split}: {len(df)-len(missing)}/{len(df)} present, "
        f"{len(missing)} missing/invalid"
    )

    return missing


if __name__ == "__main__":

    split = "train"

    csv_path = os.path.join(
        DATA_DIR,
        f"{split}.csv"
    )

    if not os.path.exists(csv_path):
        print(f"Missing CSV: {csv_path}")
        exit()


    df = pd.read_csv(csv_path)

    print(
        f"\n=== {split}: {len(df)} rows ==="
    )


    download_images(
        df,
        split,
        NUM_THREADS
    )


    missing = validate_images(
        df,
        split
    )


    print("\n=== Summary ===")

    print(
        f"{split}: "
        f"{len(df)-len(missing)}/{len(df)} images on disk "
        f"({len(missing)} missing)"
    )