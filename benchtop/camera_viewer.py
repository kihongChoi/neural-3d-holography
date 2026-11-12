"""Standalone desktop camera viewer for an SLM driven by another application.

Run: .venv\Scripts\python.exe -m benchtop.camera_viewer --start
Use --demo for an explicitly labelled, hardware-free UI check.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import tempfile
import time
import numpy as np

from PySide6.QtCore import Qt, QTimer, QLockFile, QUrl, QRect
from PySide6.QtGui import QImage, QPixmap, QShortcut, QKeySequence, QDesktopServices
from PySide6.QtWidgets import (QApplication, QDoubleSpinBox, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QMainWindow, QPlainTextEdit,
                               QPushButton, QSizePolicy, QVBoxLayout, QWidget, QMessageBox,
                               QComboBox, QFileDialog, QTabWidget, QScrollArea, QLayout)

from .viewer_service import (CameraWorker, DemoCamera, DemoLaser, DemoStage,
                             LaserWorker, Shared, SnapshotWorker, StageWorker,
                             kst_now, new_session)
from .viewer_slm import load_lut, compose_frame, prepare, sha, LUT_SHA, ROOT


class SlmOutput(QLabel):
    def __init__(self, screen, demo=False):
        super().__init__()
        self.setWindowTitle('SLM output' + (' · DEMO' if demo else ''))
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setCursor(Qt.BlankCursor)
        self.setStyleSheet('background:#808080; border:0; padding:0; margin:0;')
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(False)
        self.screen_name = screen.name()
        self.target_screen = screen
        self.demo = demo
        self.setGeometry(screen.geometry())
        self.winId()
        self.windowHandle().setScreen(screen)
        self.setGeometry(screen.geometry())

    def present(self, frame):
        image = QImage(frame.data, 1920, 1080, frame.strides[0], QImage.Format_RGB888).copy()
        self.setPixmap(QPixmap.fromImage(image))
        if not self.isVisible():
            self.windowHandle().setScreen(self.target_screen)
            self.showFullScreen()
        # On mixed-DPI desktops showFullScreen may temporarily select the
        # primary monitor. Reassert the chosen screen's geometry, not that
        # transient windowHandle().screen().
        self.setGeometry(self.target_screen.geometry())
        self.raise_()

    def verify(self, expected):
        screen = self.windowHandle().screen()
        if screen.name() != self.screen_name or screen.size().toTuple() != (1920, 1080) or screen.devicePixelRatio() != 1:
            raise RuntimeError(f'SLM 화면 연결/크기/배율 확인 실패: {screen.name()} / {screen.size().toTuple()} / {screen.devicePixelRatio()}')
        image = screen.grabWindow(0).toImage().convertToFormat(QImage.Format_RGB888)
        if image.size().toTuple() != (1920, 1080):
            raise RuntimeError('실제 SLM 프레임 크기를 확인할 수 없습니다.')
        pixels = np.frombuffer(image.constBits(), np.uint8).reshape(1080, image.bytesPerLine())[:, :1920*3].reshape(1080, 1920, 3)
        error = float(np.abs(pixels.astype(np.int16)-expected.astype(np.int16)).mean())
        if error > 1:
            raise RuntimeError(f'SLM 출력이 가려졌거나 표시 코드가 다릅니다 (평균 차이 {error:.2f}/255).')
        return error


class VideoCanvas(QLabel):
    def __init__(self):
        super().__init__('시작을 누르면 카메라 영상이 표시됩니다.\nSLM 화면은 외부 프로그램에서 표시하세요.')
        self.image = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(500, 360)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.setStyleSheet('background:#090d14; color:#94a3b8; border:1px solid #293548; border-radius:10px;')

    def show_frame(self, rgb):
        h, w, _ = rgb.shape
        self.image = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()
        self.draw()

    def draw(self):
        if self.image is not None:
            self.setPixmap(QPixmap.fromImage(self.image).scaled(self.size(), Qt.KeepAspectRatio, Qt.FastTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.draw()


class WrappedNote(QLabel):
    """Keep dynamic multi-line notes readable inside the scrollable control pane."""
    def __init__(self, text=''):
        super().__init__(text)
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)

    def fit_height(self):
        metrics = self.fontMetrics()
        bounds = metrics.boundingRect(QRect(0, 0, max(1, self.width()-8), 100000),
                                      Qt.TextWordWrap, self.text())
        height = max(bounds.height(), len(self.text().split('\n'))*metrics.lineSpacing())+8
        if height > 0 and self.minimumHeight() != height:
            self.setMinimumHeight(height)

    def setText(self, text):
        super().setText(text)
        self.fit_height()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit_height()


class Viewer(QMainWindow):
    def __init__(self, folder, demo=False):
        super().__init__()
        self.folder, self.demo = Path(folder), demo
        self.shared = Shared(folder)
        self.workers = {}
        self.closing = self.pending_laser = False
        self.seen_frame = 0
        self.last_state_write = 0
        self.stage_pending = 0.
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='slm-prepare')
        self.pattern_future = None
        self.pattern_number = 0
        self.original_codes = self.pattern_meta = self.slm_window = self.displayed_frame = None
        self.slm_state = dict(playing=False, message='파일을 선택하세요.', lut=False)
        try:
            self.lut_table = load_lut()
            self.lut_error = None
        except Exception as exc:
            self.lut_table, self.lut_error = None, str(exc)
        self.started_at = kst_now().isoformat()
        self.setWindowTitle('Bench Camera · 실시간 뷰어' + (' — DEMO / 실제 장비 아님' if demo else ''))
        self.resize(1320, 870)
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(22, 18, 22, 16)
        title = QLabel('CAMERA LIVE' + ('  ·  DEMO' if demo else ''))
        title.setStyleSheet('font-size:24px; font-weight:700; color:#eef4ff;')
        outer.addWidget(title)
        subtitle = QLabel('카메라 연속 관찰  ·  사진 / 복소 CGH → DPAC 저장·재생  ·  중앙 80% SLM 출력')
        subtitle.setStyleSheet('color:#94a3b8; margin-bottom:10px;')
        outer.addWidget(subtitle)
        body = QHBoxLayout()
        outer.addLayout(body, 1)
        left = QVBoxLayout()
        body.addLayout(left, 1)
        self.canvas = VideoCanvas()
        left.addWidget(self.canvas, 1)
        self.video_status = QLabel('대기 중')
        self.video_status.setStyleSheet('color:#66dbc1; padding:8px 0;')
        left.addWidget(self.video_status)
        self.frame_info = QLabel('카메라 25222112  ·  전체 센서 4096 × 3000')
        left.addWidget(self.frame_info)
        side = QWidget()
        side.setFixedWidth(360)
        right = QVBoxLayout(side)
        right.setContentsMargins(12, 0, 0, 0)
        body.addWidget(side)

        self.tabs = QTabWidget()
        right.addWidget(self.tabs, 1)
        panels = []
        for name in ('카메라', '스테이지', 'SLM 재생'):
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.setSizeConstraint(QLayout.SetMinimumSize)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            self.tabs.addTab(scroll, name)
            panels.append(layout)
        camera_box, camera_layout = self.group('카메라 / 레이저', panels[0])
        self.start_button = self.button('시작 · 카메라 + 레이저 ON', self.start_all, camera_layout, primary=True)
        self.stop_button = self.button('스트림 중지 · 레이저 OFF', self.stop_stream, camera_layout)
        self.laser_status = QLabel('레이저 연결 안 됨')
        self.laser_status.setWordWrap(True)
        camera_layout.addWidget(self.laser_status)
        laser_row = QHBoxLayout()
        camera_layout.addLayout(laser_row)
        self.laser_on = self.button('레이저 ON', self.turn_laser_on, laser_row)
        self.laser_off = self.button('레이저 OFF', self.turn_laser_off, laser_row)
        self.exposure = QDoubleSpinBox()
        self.exposure.setRange(20, 100000)
        self.exposure.setDecimals(1)
        self.exposure.setSingleStep(100)
        self.exposure.setValue(15000)
        self.exposure.setSuffix(' μs')
        self.exposure.setKeyboardTracking(False)
        form = QFormLayout()
        form.addRow('노출 시간', self.exposure)
        camera_layout.addLayout(form)
        self.apply_exposure = self.button('노출 적용', self.change_exposure, camera_layout)
        self.exposure_actual = QLabel('적용값 —  ·  트리거 지연 22 μs')
        camera_layout.addWidget(self.exposure_actual)

        panels[0].addStretch()
        _, stage_layout = self.group('스테이지 · LTS300/M', panels[1])
        self.position = QLabel('— mm')
        self.position.setStyleSheet('font-size:32px; font-weight:700; color:#eef4ff;')
        stage_layout.addWidget(self.position)
        self.stage_status = QLabel('연결 안 됨 · 장치 절대좌표')
        self.stage_status.setWordWrap(True)
        stage_layout.addWidget(self.stage_status)
        self.connect_stage_button = self.button('스테이지 연결', self.connect_stage, stage_layout)
        self.step = QDoubleSpinBox()
        self.step.setRange(.5, 10.)
        self.step.setSingleStep(.5)
        self.step.setDecimals(1)
        self.step.setValue(.5)
        self.step.setSuffix(' mm')
        form = QFormLayout()
        form.addRow('이동 간격', self.step)
        stage_layout.addLayout(form)
        self.speed = QDoubleSpinBox()
        self.speed.setRange(.05, 2.)
        self.speed.setSingleStep(.05)
        self.speed.setDecimals(2)
        self.speed.setValue(.5)
        self.speed.setSuffix(' mm/s')
        self.speed.setKeyboardTracking(False)
        speed_form = QFormLayout()
        speed_form.addRow('이동 속도', self.speed)
        stage_layout.addLayout(speed_form)
        self.apply_speed = self.button('속도 적용', self.change_speed, stage_layout)
        self.speed_actual = QLabel('일반 이동 설정: 0.50 mm/s')
        stage_layout.addWidget(self.speed_actual)
        move_row = QHBoxLayout()
        stage_layout.addLayout(move_row)
        self.minus = self.button('− 이동', lambda: self.move_stage(-1), move_row)
        self.plus = self.button('+ 이동', lambda: self.move_stage(1), move_row)
        self.home = self.button('호밍 · 원점 복귀', self.home_stage, stage_layout)
        self.halt = self.button('스테이지 정지  [Esc]', self.stop_stage, stage_layout)
        self.halt.setStyleSheet('background:#733345; color:#fff; border:1px solid #c45d73;')
        note = QLabel('이동 0.05–2.0 mm/s · 상한 124 mm\n호밍은 역방향 · 최대 0.5 mm/s')
        note.setStyleSheet('color:#94a3b8; font-size:11px;')
        stage_layout.addWidget(note)
        panels[1].addStretch()
        _, slm_layout = self.group('사진 / 복소 CGH → DPAC', panels[2])
        self.screen_choice = QComboBox()
        slm_layout.addWidget(self.screen_choice)
        self.screen_status = QLabel()
        self.screen_status.setWordWrap(True)
        slm_layout.addWidget(self.screen_status)
        self.refresh_screen_button = self.button('화면 연결 새로고침', self.refresh_screens, slm_layout)
        self.choose_image_button = self.button('이미지 / 복소 NPY 선택…', self.choose_image, slm_layout)
        self.image_path = None
        self.image_name = QLabel('선택한 이미지 없음')
        self.image_name.setWordWrap(True)
        slm_layout.addWidget(self.image_name)
        self.input_mode = QComboBox()
        self.input_mode.addItem('일반 사진 → 단일 패스 DPAC', 'dpac')
        self.input_mode.addItem('복소 CGH (.npy) → DPAC 저장', 'complex')
        self.input_mode.addItem('기존 원본 구동 이미지 (LUT 미적용)', 'codes')
        slm_layout.addWidget(self.input_mode)
        self.input_note = WrappedNote()
        slm_layout.addWidget(self.input_note)
        self.input_mode.currentIndexChanged.connect(self.update_input_note)
        self.update_input_note()
        self.depth = QDoubleSpinBox()
        self.depth.setRange(0, 20)
        self.depth.setDecimals(2)
        self.depth.setSingleStep(.5)
        self.depth.setValue(3.)
        self.depth.setSuffix(' mm')
        depth_form = QFormLayout()
        depth_form.addRow('설계 깊이', self.depth)
        slm_layout.addLayout(depth_form)
        self.dpac_gain = QDoubleSpinBox()
        self.dpac_gain.setRange(.01, 10.)
        self.dpac_gain.setDecimals(2)
        self.dpac_gain.setSingleStep(.1)
        self.dpac_gain.setValue(1.)
        gain_form = QFormLayout()
        gain_form.addRow('복소장 DPAC gain', self.dpac_gain)
        slm_layout.addLayout(gain_form)
        self.generate_button = self.button('홀로그램 생성 / 준비', self.prepare_image, slm_layout, primary=True)
        self.pattern_status = WrappedNote('사진 비율 유지 · 중앙 1728 × 960\n전체 SLM 면적의 80% · 주변 코드 128')
        slm_layout.addWidget(self.pattern_status)
        self.open_pattern_button = self.button('생성 파일 저장 폴더 열기', self.open_pattern_folder, slm_layout)
        self.lut_toggle = QPushButton('LUT OFF · 항등 구동')
        self.lut_toggle.setCheckable(True)
        self.lut_toggle.toggled.connect(self.toggle_lut)
        slm_layout.addWidget(self.lut_toggle)
        lut_note = QLabel('11월 11일 실물 검증 RGB LUT\n동일 원본 코드에 한 번 적용')
        lut_note.setWordWrap(True)
        if self.lut_error:
            lut_note.setText(self.lut_error)
        slm_layout.addWidget(lut_note)
        self.play_button = self.button('선택 화면에 SLM 재생', self.play_slm, slm_layout)
        self.release_button = self.button('SLM 재생 해제', self.stop_slm, slm_layout)
        self.output_status = QLabel('재생 대기 · 외부 프로그램 화면 유지')
        self.output_status.setWordWrap(True)
        slm_layout.addWidget(self.output_status)
        panels[2].addStretch()
        self.refresh_screens()
        QApplication.instance().screenAdded.connect(self.refresh_screens)
        QApplication.instance().screenRemoved.connect(self.screen_removed)
        self.save = self.button('현재 프레임 PNG 저장', self.save_frame, right)
        folder_label = QLabel(str(self.folder))
        folder_label.setWordWrap(True)
        folder_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        folder_label.setStyleSheet('color:#94a3b8; font-size:10px;')
        right.addWidget(folder_label)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(150)
        self.log.setMaximumHeight(110)
        outer.addWidget(self.log)
        self.statusBar().showMessage('영상은 자동 저장되지 않습니다. PNG 저장 시 전체 프레임 한 장을 저장합니다.')
        self.setStyleSheet('''
            QMainWindow, QWidget { background:#141c29; color:#d8e2f1; font-family:"Malgun Gothic"; font-size:12px; }
            QGroupBox { border:1px solid #334155; border-radius:9px; margin-top:12px; padding:12px 10px 10px; }
            QGroupBox::title { subcontrol-origin:margin; left:12px; padding:0 6px; color:#aebfd6; }
            QPushButton { background:#26354b; border:1px solid #40516b; border-radius:6px; padding:9px; }
            QPushButton:hover { background:#334864; }
            QPushButton:disabled { color:#64748b; background:#1c2738; border-color:#293548; }
            QDoubleSpinBox { background:#0c1421; padding:7px; border:1px solid #40516b; border-radius:5px; }
            QPlainTextEdit { background:#0c1421; border:1px solid #293548; border-radius:6px; font-size:11px; }
            QTabWidget::pane { border:0; }
            QTabBar::tab { background:#1c2738; padding:9px 12px; border-radius:4px; }
            QTabBar::tab:selected { background:#263f52; color:#66dbc1; }
            QComboBox { background:#0c1421; padding:7px; border:1px solid #40516b; border-radius:5px; }
            QPushButton:checked { background:#166c65; color:white; border-color:#66dbc1; }
        ''')
        self.escape = QShortcut(QKeySequence('Esc'), self)
        self.escape.activated.connect(self.stop_stage)
        writer = SnapshotWorker(self.shared)
        self.workers['snapshot'] = writer
        writer.start()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(50)
        self.shared.log('viewer', 'started', demo=demo)

    def refresh_screens(self, *_):
        old = self.screen_choice.currentData()
        self.screen_choice.clear()
        app = QApplication.instance()
        screens = [s for s in app.screens() if s != app.primaryScreen()]
        for s in screens:
            self.screen_choice.addItem(f'{s.name().strip()} · {s.size().width()}×{s.size().height()} · {s.devicePixelRatio()*100:g}%', s.name())
        index = self.screen_choice.findData(old)
        if index >= 0:
            self.screen_choice.setCurrentIndex(index)
        else:
            preferred = self.screen_choice.findData('LETO-3-0088')
            if preferred >= 0:
                self.screen_choice.setCurrentIndex(preferred)
        self.screen_status.setText('보조화면을 선택하세요. 1920×1080 / 100% 필요.' if screens else 'SLM 보조화면 미연결')

    def screen_removed(self, screen):
        if self.slm_window and self.slm_window.screen_name == screen.name():
            self.stop_slm()
            self.slm_state['message'] = 'SLM 연결이 해제되었습니다.'
        self.refresh_screens()

    def choose_image(self):
        path, _ = QFileDialog.getOpenFileName(self, 'SLM 이미지 / 복소 CGH 선택', str(ROOT if self.image_path is None else Path(self.image_path).parent),
                                            'SLM inputs (*.npy *.png *.jpg *.jpeg *.bmp *.tif *.tiff);;Complex CGH (*.npy);;Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)')
        if path:
            if self.slm_state['playing']:
                self.stop_slm()
            self.original_codes = self.pattern_meta = None
            self.image_path = path
            self.image_name.setText(Path(path).name)
            if Path(path).suffix.lower() == '.npy':
                self.input_mode.setCurrentIndex(self.input_mode.findData('complex'))
            elif self.input_mode.currentData() == 'complex':
                self.input_mode.setCurrentIndex(self.input_mode.findData('dpac'))
            self.pattern_status.setText('파일 선택 완료 · 홀로그램 생성 버튼을 눌러주세요.')

    def update_input_note(self):
        if self.input_mode.currentData() == 'complex':
            text = ('SLM 면 complex64/128 · CHW/HWC RGB 또는 2D\n'
                    '단일 채널은 RGB 복제 · 공통 진폭 정규화\n'
                    '1728×960 이하: 0 패딩 / 1920×1080: 중앙 ROI\n'
                    '리사이즈·추가 전파 없음 · PNG 자동 저장(LUT OFF)')
        else:
            text = '사진: 비율 유지 후 전파 계산\n기존 구동 이미지: LUT 미적용 RGB8, 픽셀 크기 유지'
        self.input_note.setText(text)

    def open_pattern_folder(self):
        if self.pattern_meta:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.pattern_meta['output_directory']))

    def prepare_image(self):
        if not self.image_path or self.pattern_future is not None:
            return
        self.pattern_number += 1
        self.pattern_future = self.executor.submit(prepare, self.image_path, self.input_mode.currentData(),
                                                  self.depth.value(), self.folder/'patterns'/f'{self.pattern_number:04}',
                                                  self.dpac_gain.value())
        self.pattern_status.setText('홀로그램 준비 중… 카메라는 계속 수신합니다.')

    def toggle_lut(self, enabled):
        self.lut_toggle.setText('LUT ON · 검증 보정표' if enabled else 'LUT OFF · 항등 구동')
        if self.slm_state['playing']:
            self.play_slm()

    def play_slm(self):
        if self.original_codes is None:
            return
        try:
            if self.lut_toggle.isChecked() and self.lut_table is None:
                raise RuntimeError(self.lut_error)
            frame = compose_frame(self.original_codes, self.lut_table if self.lut_toggle.isChecked() else None)
            if not self.demo:
                app = QApplication.instance()
                screen = next((s for s in app.screens() if s.name() == self.screen_choice.currentData()), None)
                if screen is None or screen == app.primaryScreen():
                    raise RuntimeError('SLM 보조 디스플레이를 선택하세요.')
                if screen.size().toTuple() != (1920, 1080) or screen.devicePixelRatio() != 1:
                    raise RuntimeError('SLM은 1920×1080 및 Windows 배율 100%가 필요합니다.')
                if self.slm_window and self.slm_window.screen_name != screen.name():
                    self.stop_slm()
                if self.slm_window is None:
                    self.slm_window = SlmOutput(screen)
                self.slm_window.present(frame)
            self.displayed_frame = frame
            self.slm_state.update(playing=True, message='DEMO · 출력 코드 준비' if self.demo else 'SLM 출력 중 · 확인 중',
                                  screen=self.screen_choice.currentData(), lut=self.lut_toggle.isChecked(),
                                  original_codes_sha256=sha(self.original_codes.tobytes()), displayed_sha256=sha(frame.tobytes()),
                                  lut_sha256=LUT_SHA if self.lut_toggle.isChecked() else None,
                                  pattern=self.pattern_meta)
            self.shared.log('slm', 'presented', state=self.slm_state.copy())
            if not self.demo:
                QTimer.singleShot(300, self.verify_slm)
        except Exception as exc:
            self.slm_state['message'] = str(exc)
            self.shared.log('slm', str(exc), error=True)

    def verify_slm(self):
        if not self.slm_window or self.closing:
            return
        try:
            error = self.slm_window.verify(self.displayed_frame)
            self.slm_state.update(message='SLM 재생 중 · 실제 화면 확인됨', framebuffer_error=error)
        except Exception as exc:
            self.stop_slm()
            self.slm_state['message'] = str(exc)
            self.shared.log('slm', str(exc), error=True)

    def stop_slm(self):
        if self.slm_window:
            self.slm_window.close()
            self.slm_window.deleteLater()
            self.slm_window = None
        self.displayed_frame = None
        self.slm_state.update(playing=False, message='재생 해제 · 외부 프로그램 화면 유지')
        self.shared.log('slm', 'released')

    @staticmethod
    def group(title, parent):
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        layout.setSizeConstraint(QLayout.SetMinimumSize)
        parent.addWidget(box)
        return box, layout

    @staticmethod
    def button(title, action, layout, primary=False):
        button = QPushButton(title)
        button.clicked.connect(action)
        if primary:
            button.setStyleSheet('background:#166c65; color:#fff; font-weight:700;')
        layout.addWidget(button)
        return button

    def alive(self, name):
        return name in self.workers and self.workers[name].is_alive()

    def submit(self, name, *command):
        if not self.alive(name) or not self.workers[name].submit(*command):
            self.statusBar().showMessage('연결 상태를 확인하거나 이전 명령이 끝난 뒤 다시 눌러주세요.', 4000)
            return False
        return True

    def start_all(self):
        if self.closing:
            return
        if not self.alive('camera'):
            args = dict(factory=DemoCamera) if self.demo else {}
            self.workers['camera'] = CameraWorker(self.shared, exposure=self.exposure.value(), **args)
            self.workers['camera'].start()
        if not self.alive('laser'):
            args = dict(factory=DemoLaser) if self.demo else {}
            self.workers['laser'] = LaserWorker(self.shared, **args)
            self.workers['laser'].start()
        self.pending_laser = True
        self.connect_stage()

    def connect_stage(self):
        if not self.alive('stage') and not self.closing:
            args = dict(factory=DemoStage) if self.demo else {}
            self.workers['stage'] = StageWorker(self.shared, **args)
            self.workers['stage'].start()

    def turn_laser_on(self):
        self.pending_laser = False
        self.submit('laser', 'on')

    def turn_laser_off(self):
        self.pending_laser = False
        if self.alive('laser'):
            self.workers['laser'].standby()

    def stop_stream(self):
        self.turn_laser_off()
        if self.alive('camera'):
            self.workers['camera'].quit.set()

    def change_exposure(self):
        self.submit('camera', 'exposure', self.exposure.value())

    def change_speed(self):
        if self.submit('stage', 'speed', self.speed.value()):
            self.stage_pending = time.monotonic()+1

    def move_stage(self, direction):
        if self.submit('stage', 'step', self.step.value(), direction):
            self.stage_pending = time.monotonic()+1

    def home_stage(self):
        if self.submit('stage', 'home'):
            self.stage_pending = time.monotonic()+1

    def stop_stage(self):
        if self.alive('stage'):
            self.workers['stage'].stop()

    def save_frame(self):
        frame = self.shared.get_frame()
        if frame is not None and time.monotonic()-frame[2]['received'] < 2:
            self.submit('snapshot', 'save', frame[1], frame[2])

    def tick(self):
        if self.pattern_future is not None and self.pattern_future.done():
            try:
                codes, meta = self.pattern_future.result()
                if not self.closing:
                    if self.slm_state['playing']:
                        self.stop_slm()
                    self.original_codes, self.pattern_meta = codes, meta
                    description = f"DPAC · 설계 {meta['design_mm']:g} mm" if meta['mode'] == 'single_pass_dpac' else '원본 구동 코드'
                    if meta['mode'] == 'complex_dpac':
                        clipping = ' / '.join(f'{100*v:.2f}%' for v in meta['clipped'])
                        description = (f"복소 DPAC · gain {meta['gain']:g} · 공통 max {meta['amp_max']:.4g}\n"
                                       f"클리핑 R/G/B: {clipping}\n1920×1080 dpac_slm.png 저장됨 (LUT OFF)")
                    self.pattern_status.setText(f"준비됨: {Path(meta['source']).name}\n{description}\n저장: patterns/{Path(meta['output_directory']).name}")
                    self.shared.log('slm', 'pattern_prepared', pattern=meta)
            except Exception as exc:
                self.pattern_status.setText(f'준비 실패: {exc}')
                self.shared.log('slm', str(exc), error=True)
            self.pattern_future = None
        state = self.shared.snapshot()
        camera, stage, laser = (state[n] for n in ('camera', 'stage', 'laser'))
        frame = self.shared.get_frame()
        fresh = bool(camera['connected'] and frame and time.monotonic()-frame[2]['received'] < 2)
        if frame and frame[0] != self.seen_frame:
            self.seen_frame = frame[0]
            self.canvas.show_frame(frame[1])
        self.video_status.setText(('● LIVE  ·  ' if fresh else '● 영상 갱신 없음  ·  ') + camera['message'])
        self.video_status.setStyleSheet('color:' + ('#66dbc1' if fresh else '#f3bc6b') + '; padding:8px 0;')
        if frame:
            h, w = frame[1].shape[:2]
            age = time.monotonic()-frame[2]['received']
            self.frame_info.setText(f"{w} × {h}  ·  {camera.get('fps', 0):.1f} FPS  ·  프레임 {frame[2]['frame_id']}  ·  마지막 수신 {age:.1f}초 전")
        if 'exposure_us' in camera:
            self.exposure.setRange(camera['exposure_min'], camera['exposure_max'])
            self.exposure_actual.setText(f"적용값 {camera['exposure_us']:,.1f} μs  ·  지연 {camera['delay_us']:g} μs")
        self.laser_status.setText('레이저: ' + laser['message'])
        if laser.get('on') is None and laser.get('error'):
            self.laser_status.setStyleSheet('color:#ff8294;')
        else:
            self.laser_status.setStyleSheet('color:' + ('#66dbc1' if laser.get('on') else '#aebfd6') + ';')
        if 'position_mm' in stage:
            self.position.setText(f"{stage['position_mm']:.3f} mm" if stage['connected'] else '— mm')
        self.stage_status.setText(stage.get('error') or (stage['message'] + ' · ' +
                                  ('HOMED' if stage.get('homed') else '호밍 필요')))
        enabled = not self.closing
        ready = enabled and stage['connected'] and not stage.get('moving') and not stage.get('operation') and time.monotonic() > self.stage_pending
        self.home.setEnabled(ready)
        self.apply_speed.setEnabled(ready)
        self.speed.setEnabled(ready)
        if 'speed_mm_s' in stage:
            self.speed_actual.setText(f"일반 이동 설정: {stage['speed_mm_s']:.2f} mm/s")
        self.minus.setEnabled(ready and stage.get('homed', False) and stage.get('enabled', False))
        self.plus.setEnabled(self.minus.isEnabled())
        self.halt.setEnabled(enabled and stage['connected'])
        self.connect_stage_button.setEnabled(enabled and not self.alive('stage'))
        self.start_button.setEnabled(enabled and (not self.alive('camera') or not self.alive('laser')))
        self.stop_button.setEnabled(enabled and self.alive('camera'))
        self.apply_exposure.setEnabled(enabled and camera['connected'])
        self.laser_on.setEnabled(enabled and fresh and laser['connected'] and not laser.get('on'))
        self.laser_off.setEnabled(enabled and laser['connected'])
        self.save.setEnabled(enabled and fresh)
        preparing = self.pattern_future is not None
        self.choose_image_button.setEnabled(enabled and not preparing)
        self.generate_button.setEnabled(enabled and bool(self.image_path) and not preparing)
        self.input_mode.setEnabled(enabled and not preparing)
        self.depth.setEnabled(enabled and not preparing and self.input_mode.currentData() == 'dpac')
        self.dpac_gain.setEnabled(enabled and not preparing and self.input_mode.currentData() == 'complex')
        self.open_pattern_button.setEnabled(enabled and self.pattern_meta is not None and not preparing)
        self.lut_toggle.setEnabled(enabled and self.lut_table is not None and self.original_codes is not None and not preparing)
        self.play_button.setEnabled(enabled and self.original_codes is not None and not preparing and (self.demo or self.screen_choice.count() > 0))
        self.release_button.setEnabled(enabled and self.slm_state['playing'])
        self.screen_choice.setEnabled(enabled and not self.slm_state['playing'])
        self.output_status.setText(self.slm_state['message'])
        if self.pending_laser and fresh and laser['connected']:
            self.turn_laser_on()
        if self.pending_laser and not self.alive('camera'):
            self.pending_laser = False
        while not self.shared.messages.empty():
            row = self.shared.messages.get_nowait()
            suffix = (' · '+row['path']) if 'path' in row else ''
            self.log.appendPlainText(f"{row['time_kst'][11:19]} [{row['device']}] {row['message']}{suffix}")
        if time.monotonic()-self.last_state_write > 1:
            if self.slm_window:
                screen = self.slm_window.windowHandle().screen()
                if screen.name() != self.slm_window.screen_name or screen.size().toTuple() != (1920, 1080) or screen.devicePixelRatio() != 1:
                    self.stop_slm()
                    self.slm_state['message'] = '디스플레이 연결/배율 변경으로 재생 해제'
            self.write_state(state)
            self.last_state_write = time.monotonic()
        if self.closing and self.pattern_future is None and not any(w.is_alive() for w in self.workers.values()):
            self.write_state(self.shared.snapshot(), closed=True)
            self.timer.stop()
            self.close()

    def write_state(self, state, closed=False):
        payload = dict(started_at=self.started_at, updated_at=kst_now().isoformat(), demo=self.demo,
                       closed=closed, closing=self.closing, frames_received=self.shared.sequence,
                       devices=state, session=str(self.folder), slm=self.slm_state,
                       displays=[dict(name=s.name(), width=s.size().width(), height=s.size().height(), scale=s.devicePixelRatio())
                                 for s in QApplication.instance().screens()])
        try:
            temporary = self.folder/'state.tmp'
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
            temporary.replace(self.folder/'state.json')
        except OSError as exc:
            self.statusBar().showMessage(f'상태 기록 오류: {exc}')

    def closeEvent(self, event):
        if self.pattern_future is not None or any(w.is_alive() for w in self.workers.values()):
            event.ignore()
            if not self.closing:
                self.closing = True
                self.pending_laser = False
                self.stop_stage()
                if self.slm_state['playing']:
                    self.stop_slm()
                for worker in self.workers.values():
                    worker.quit.set()
                self.statusBar().showMessage('종료 중: 스테이지 정지 / 레이저 OFF 확인 / 카메라 설정 복원…')
        else:
            self.executor.shutdown(wait=False, cancel_futures=True)
            errors = {k: v.get('error') for k, v in self.shared.snapshot().items() if v.get('error')}
            self.shared.log('viewer', 'closed', errors=errors)
            self.write_state(self.shared.snapshot(), closed=True)
            laser = self.shared.snapshot()['laser']
            if laser.get('on') is None and laser.get('error'):
                QMessageBox.warning(self, '레이저 종료 상태 확인 필요',
                                    '레이저 OFF readback에 실패했습니다.\n' + laser['error'])
            event.accept()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', action='store_true', help='Connect devices and enable externally synchronized laser')
    parser.add_argument('--demo', action='store_true', help='Synthetic UI only; no hardware access')
    parser.add_argument('--results', type=Path, default=Path(__file__).resolve().parents[1]/'results')
    parser.add_argument('--smoke-test', action='store_true', help='Demo-only automatic UI check and exit')
    args = parser.parse_args()
    if args.smoke_test and not args.demo:
        parser.error('--smoke-test requires --demo')
    app = QApplication(sys.argv[:1])
    app.setApplicationName('Bench Camera Viewer')
    lock = QLockFile(str(Path(tempfile.gettempdir()) /
                        ('neural3d-camera-viewer-demo.lock' if args.demo else 'neural3d-camera-viewer.lock')))
    lock.setStaleLockTime(0)  # Running PIDs stay exclusive; dead-process locks can be recovered.
    if not lock.tryLock(0):
        QMessageBox.information(None, '카메라 뷰어', '이미 실행 중인 카메라 뷰어 창을 사용하세요.')
        return 1
    folder = new_session(args.results, args.demo)
    window = Viewer(folder, args.demo)
    # Never place this control window on the physical SLM monitor.
    screen = next((s for s in app.screens() if s.name() != 'LETO-3-0088'), None)
    if screen is None:
        window.close()
        raise RuntimeError('SLM 이외의 조작용 디스플레이가 필요합니다.')
    area = screen.availableGeometry()
    window.resize(min(1320, area.width()-40), min(870, area.height()-40))
    window.move(area.x()+20, area.y()+20)
    window.show()
    print(folder, flush=True)
    if args.start or args.smoke_test:
        QTimer.singleShot(100, window.start_all)
    if args.smoke_test:
        QTimer.singleShot(1800, lambda: (window.exposure.setValue(8000), window.change_exposure()))
        QTimer.singleShot(2400, lambda: window.move_stage(-1))
        QTimer.singleShot(3000, window.stop_stage)
        QTimer.singleShot(3500, lambda: window.grab().save(str(folder/'demo-ui.png')))
        QTimer.singleShot(4200, window.close)
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
