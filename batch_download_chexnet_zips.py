# Download the 56 zip files in Images_png in batches
import time
import datetime
import sys
import shutil
from tqdm import tqdm
from urllib import request

# URLs for the zip files
links = [
    'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
    'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
    'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
    'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
    'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
    'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
    'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
    'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
    'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
    'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
    'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
    'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
]

# list of downloaded files
files = [
    'images_01.tar.gz',
    'images_02.tar.gz',
    'images_03.tar.gz',
    'images_04.tar.gz',
    'images_05.tar.gz',
    'images_06.tar.gz',
    'images_07.tar.gz',
    'images_08.tar.gz',
    'images_09.tar.gz',
    'images_10.tar.gz',
    'images_11.tar.gz',
    'images_12.tar.gz',
]


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    elapsed_time = str(datetime.timedelta(minutes=duration))
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * (int(duration) + 1))) #add: +1
    percent = min(int(count * block_size * 100 / total_size), 100)
    sys.stdout.write("\r.........%d%%, %d MB, %d KB/s, %s elapsed" %
                    (percent, progress_size / (1024 * 1024), speed, elapsed_time))
    sys.stdout.flush()

def download_files():
    for idx, link in enumerate(links):
        fn = 'images_%02d.tar.gz' % (idx+1)
        print('\ndownloading', fn, '...')
        request.urlretrieve(link, fn, reporthook)  # download the zip file

    print ("Download complete.")

    return True


def uncompress_files():
    print ("Uncompressing files")

    extract_path = './images'

    for filename in tqdm(files):
        shutil.unpack_archive(filename)

    print ('Extracted all files under: {0}'.format(extract_path))


def main():
    response = download_files()

    if response == True:
        uncompress_files()


if __name__ == '__main__':
    main()
