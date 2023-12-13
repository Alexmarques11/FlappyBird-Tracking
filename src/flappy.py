import asyncio
import sys
import cv2
import pygame
import threading
import numpy as np
import time
from pygame.locals import K_ESCAPE, K_SPACE, K_UP, KEYDOWN, QUIT

from .entities import (
    Background,
    Floor,
    GameOver,
    Pipes,
    Player,
    PlayerMode,
    Score,
    WelcomeMessage,
)
from .utils import GameConfig, Images, Sounds, Window


class Flappy:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Flappy Bird")
        window = Window(288, 512)
        screen = pygame.display.set_mode((window.width, window.height))
        images = Images()

        self.config = GameConfig(
            screen=screen,
            clock=pygame.time.Clock(),
            fps=30,
            window=window,
            images=images,
            sounds=Sounds(),
        )
        self.cap = cv2.VideoCapture(0)
        self.tracker = cv2.TrackerKCF_create()
        self.tracking = False

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    async def start(self):
        threading.Thread(target=self.tracking_thread, daemon=True).start()

        while True:
            self.background = Background(self.config)
            self.floor = Floor(self.config)
            self.player = Player(self.config)
            self.welcome_message = WelcomeMessage(self.config)
            self.game_over_message = GameOver(self.config)
            self.pipes = Pipes(self.config)
            self.score = Score(self.config)

            await self.splash()
            await self.play()
            await self.game_over()

    async def splash(self):
        """Shows welcome splash screen animation of flappy bird"""

        self.player.set_mode(PlayerMode.SHM)

        while True:
            for event in pygame.event.get():
                self.check_quit_event(event)
                if self.is_tap_event(event):
                    return

            self.background.tick()
            self.floor.tick()
            self.player.tick()
            self.welcome_message.tick()

            pygame.display.update()
            await asyncio.sleep(0)
            self.config.tick()

    def check_quit_event(self, event):
        if event.type == QUIT or (
                event.type == KEYDOWN and event.key == K_ESCAPE
        ):
            pygame.quit()
            sys.exit()

    def is_tap_event(self, event):
        m_left, _, _ = pygame.mouse.get_pressed()
        space_or_up = event.type == KEYDOWN and (
                event.key == K_SPACE or event.key == K_UP
        )
        screen_tap = event.type == pygame.FINGERDOWN
        return m_left or space_or_up or screen_tap

    async def play(self):
        self.score.reset()
        self.player.set_mode(PlayerMode.NORMAL)

        while True:
            if self.player.collided(self.pipes, self.floor):
                return

            for i, pipe in enumerate(self.pipes.upper):
                if self.player.crossed(pipe):
                    self.score.add()

            for event in pygame.event.get():
                self.check_quit_event(event)
                if self.is_tap_event(event):
                    self.player.flap()

            self.background.tick()
            self.floor.tick()
            self.pipes.tick()
            self.score.tick()

            # Obtenha o frame da câmera
            _, frame = self.cap.read()

            # Detecte a posição do rosto
            face_center = self.detect_face(frame)

            if face_center is not None and self.tracking:
                x, y = face_center
                self.player.update_position(y)

            self.player.tick()

            pygame.display.update()
            await asyncio.sleep(0)
            self.config.tick()

    async def game_over(self):
        """crashes the player down and shows gameover image"""

        self.player.set_mode(PlayerMode.CRASH)
        self.pipes.stop()
        self.floor.stop()

        while True:
            for event in pygame.event.get():
                self.check_quit_event(event)
                if self.is_tap_event(event):
                    if self.player.y + self.player.h >= self.floor.y - 1:
                        return

            self.background.tick()
            self.floor.tick()
            self.pipes.tick()
            self.score.tick()
            self.player.tick()
            self.game_over_message.tick()

            self.config.tick()
            pygame.display.update()
            await asyncio.sleep(0)

    def tracking_thread(self):
        cv2.namedWindow("Image")
        last_frame_timestamp = time.time()
        bbox = None

        while True:
            begin_time_stamp = time.time()

            try:
                framerate = 1 / (begin_time_stamp - last_frame_timestamp)
            except ZeroDivisionError:
                framerate = 0

            last_frame_timestamp = begin_time_stamp

            if not self.cap.isOpened():
                self.cap.open(0)
            _, frame = self.cap.read()

            face_center = self.detect_face(frame)
            if face_center is not None and bbox is None:
                bbox = (face_center[0] - 20, face_center[1] - 40, 40, 80)
                self.tracker.init(frame, bbox)
                self.tracking = True

            if self.tracking:
                if bbox is not None:
                    track_ok, bbox = self.tracker.update(frame)
                    if track_ok:
                        x, y, w, h = map(int, bbox)
                        self.player.update_position(y)

            text_to_show = str(int(np.round(framerate))) + " fps"
            cv2.putText(img=frame,
                        text=text_to_show,
                        org=(5, 15),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(0, 255, 0),
                        thickness=1)

            cv2.imshow(winname="Image", mat=frame)

            c = cv2.waitKey(delay=1)
            if c == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def detect_face(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            return x + w // 2, y + h // 2

        return None

    def __del__(self):
        self.cap.release()

