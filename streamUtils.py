

import glob
import hashlib
import json
import os
import random
import shutil
import time
from threading import Thread


class LoadStreams:
    # YOLOv5 streamloader, i.e. `python detect.py --source 'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP streams`
    def __init__(self, sources='streams.txt', img_size=640, stride=32, auto=True):
        ##   for path, im, im0s, vid_cap, s in dataset:
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride

        if os.path.isfile(sources):
            with open(sources) as f:
                sources = [x.strip() for x in f.read(
                ).strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs, self.fps, self.frames, self.threads = [
            None] * n, [0] * n, [0] * n, [None] * n
        # clean source names for later
        self.sources = [clean_str(x) for x in sources]
        self.auto = auto
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            st = f'{i + 1}/{n}: {s}... '
            if 'youtube.com/' in s or 'youtu.be/' in s:  # if source is YouTube video
                check_requirements(('pafy', 'youtube_dl'))
                import pafy
                s = pafy.new(s).getbest(preftype="mp4").url  # YouTube URL
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f'{st}Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps[i] = max(cap.get(cv2.CAP_PROP_FPS) %
                              100, 0) or 30.0  # 30 FPS fallback
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float(
                'inf')  # infinite stream fallback

            _, self.imgs[i] = cap.read()  # guarantee first frame
            self.threads[i] = Thread(
                target=self.update, args=([i, cap, s]), daemon=True)
            LOGGER.info(
                f"{st} Success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")
            self.threads[i].start()
        LOGGER.info('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[
                     0].shape for x in self.imgs])
        # rect inference if all shapes equal
        self.rect = np.unique(s, axis=0).shape[0] == 1
        if not self.rect:
            LOGGER.warning(
                'WARNING: Stream shapes differ. For optimal performance supply similarly-shaped streams.')

    def update(self, i, cap, stream):
        # Read stream `i` frames in daemon thread
        # frame number, frame array, inference every 'read' frame
        n, f, read = 0, self.frames[i], 1
        while cap.isOpened() and n < f:
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n % read == 0:
                success, im = cap.retrieve()
                if success:
                    self.imgs[i] = im
                else:
                    LOGGER.warning(
                        'WARNING: Video stream unresponsive, please check your IP camera connection.')
                    self.imgs[i] *= 0
                    cap.open(stream)  # re-open stream if signal was lost
            time.sleep(1 / self.fps[i])  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img0 = self.imgs.copy()
        img = [letterbox(x, self.img_size, stride=self.stride,
                         auto=self.rect and self.auto)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        # BGR to RGB, BHWC to BCHW
        img = img[..., ::-1].transpose((0, 3, 1, 2))
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None, ''

    def __len__(self):
        # 1E12 frames = 32 streams at 30 FPS for 30 years
        return len(self.sources)
