#!/usr/bin/env python3
"""
ImageNet-1K Dataset Downloader from Hugging Face Hub (ILSVRC/imagenet-1k).

Downloads all ~336 Parquet shards (~140-150 GB) with multi-threading,
automatic resume capability, and local cache management.
"""

import os
import sys
import time
import argparse
import huggingface_hub
from huggingface_hub import HfApi, snapshot_download

def parse_args():
    parser = argparse.ArgumentParser(description="Download ImageNet-1K from Hugging Face Hub.")
    parser.add_argument(
        '--repo_id',
        type=str,
        default='ILSVRC/imagenet-1k',
        help="Hugging Face dataset repository ID (default: ILSVRC/imagenet-1k)"
    )
    parser.add_argument(
        '--local_dir',
        type=str,
        default=None,
        help="Custom target directory to store dataset. If omitted, downloads into standard Hugging Face cache (~/.cache/huggingface/hub/)"
    )
    parser.add_argument(
        '--hf_token',
        type=str,
        default=None,
        help="Hugging Face user access token (defaults to stored token or HF_TOKEN env var)"
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=8,
        help="Number of concurrent download threads (default: 8)"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Resolve HF Token
    token = args.hf_token or os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    if not token:
        try:
            token = huggingface_hub.get_token()
        except Exception:
            token = None

    print("=" * 75)
    print(f"📦 ImageNet-1K Downloader via Hugging Face Hub")
    print(f"Repository : {args.repo_id}")
    if args.local_dir:
        print(f"Target Dir : {os.path.abspath(args.local_dir)}")
    else:
        print(f"Target Dir : Default Hugging Face Cache (~/.cache/huggingface/hub/)")
    print(f"Workers    : {args.max_workers} concurrent threads")
    print("=" * 75)

    if not token:
        print("\n❌ Error: Hugging Face Token tidak ditemukan!")
        print("Dataset ILSVRC/imagenet-1k memerlukan otentikasi akun Hugging Face.")
        print("Silakan jalankan: python -c 'import huggingface_hub; huggingface_hub.login()'")
        print("Atau berikan token via parameter: --hf_token <YOUR_HF_TOKEN>")
        sys.exit(1)

    # 2. Check Gated Repo Access
    api = HfApi(token=token)
    user_name = "User"
    try:
        user_info = api.whoami()
        user_name = user_info.get('name', 'User')
        print(f"✅ Otentikasi token berhasil (User: {user_name})")
        # Check actual file access in gated repository
        api.hf_hub_download(repo_id=args.repo_id, filename='.gitattributes', repo_type='dataset')
        print(f"✅ Akses repository disetujui (Access Granted)")
    except Exception as e:
        err_msg = str(e)
        if "403" in err_msg or "401" in err_msg or "GatedRepoError" in err_msg or "restricted" in err_msg.lower() or "not in the authorized list" in err_msg.lower():
            print("\n⚠️  AKSES DATASET DIBATASI (Gated Dataset)!")
            print("-" * 75)
            print(f"Akun Hugging Face '{user_name}' belum menyetujui lisensi (Terms of Use).")
            print("Langkah mudah:")
            print("1. Buka browser dan kunjungi: https://huggingface.co/datasets/ILSVRC/imagenet-1k")
            print("2. Login dengan akun yang sama.")
            print("3. Di bagian atas halaman, klik tombol 'Agree and access repository' (atau 'Acknowledge terms').")
            print("4. Persetujuan bersifat instan. Begitu diklik, jalankan script ini kembali.")
            print("-" * 75)
            sys.exit(1)
        else:
            print(f"❌ Terjadi kesalahan saat memeriksa repository: {e}")
            sys.exit(1)

    # 3. Start Multi-Threaded Resilient Download
    print(f"\nMemulai pengunduhan dataset (~140-150 GB)...")
    print("Proses ini mendukung RESUME otomatis jika koneksi terputus.")
    t0 = time.perf_counter()

    try:
        download_kwargs = {
            "repo_id": args.repo_id,
            "repo_type": "dataset",
            "token": token,
            "max_workers": args.max_workers,
            "resume_download": True,
        }
        if args.local_dir:
            download_kwargs["local_dir"] = args.local_dir
            download_kwargs["local_dir_use_symlinks"] = False

        downloaded_path = snapshot_download(**download_kwargs)
        duration_sec = time.perf_counter() - t0
        duration_min = duration_sec / 60.0

        print("\n" + "=" * 75)
        print(f"🎉 Pengunduhan ImageNet-1K Selesai dalam {duration_min:.1f} menit!")
        print(f"Lokasi Dataset: {downloaded_path}")
        print("=" * 75)

    except KeyboardInterrupt:
        print("\n\n⚠️  Pengunduhan dihentikan oleh pengguna.")
        print("Anda dapat menjalankan kembali script ini kapan saja untuk melanjutkan (resume) tanpa mengulang dari awal.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Terjadi kesalahan: {e}")
        print("Jika error 401 Gated Repo: Buka https://huggingface.co/datasets/ILSVRC/imagenet-1k di browser dan klik 'Agree and access repository'.")
        sys.exit(1)

if __name__ == '__main__':
    main()
