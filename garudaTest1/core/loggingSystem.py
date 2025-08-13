"""
loggingSystem.py

Provides the `EventLogger` class to manage event logging for a detection system.
This includes writing logs to text files, saving image captures of events, and
overlaying recent event information on video frames.

Requirements
------------
- OpenCV
- Python standard library: os, datetime
- Project configuration variables from `config`:
    - LOGS_DIR
    - CAPTURES_DIR
    - MAX_LOGS_DISPLAY
    - LOG_AREA_OFFSET_Y
"""
import os
import cv2
import datetime
from config import LOGS_DIR, CAPTURES_DIR, MAX_LOGS_DISPLAY, LOG_AREA_OFFSET_Y

class EventLogger:
    """
    Handles event logging, saving of related image captures, and drawing event
    history on video frames.

    Attributes
    ----------
    events : list of str
        Stores the most recent event messages in memory.

    Methods
    -------
    log_event(objeto, zona_idx, frame_final=None)
        Records a new event, saves it to a text log, and optionally saves a frame capture.
    draw_on_frame(frame)
        Draws the most recent event messages on the given frame.
    """
    def __init__(self):
        self.events = []

    def log_event(self, objeto, zona_idx, frame_final=None):
        """
        Records a new intrusion or detection event.

        Parameters
        ----------
        objeto : str
            The label of the detected object.
        zona_idx : int
            Index of the restricted zone where the event occurred (0-based).
        frame_final : numpy.ndarray, optional
            Final combined frame to save as an image capture.

        Notes
        -----
        - The event is appended to the in-memory event list.
        - A log entry is saved in `eventos.txt` inside `LOGS_DIR`.
        - If a frame is provided, an image capture is saved in `CAPTURES_DIR`.
        - Keeps a maximum of 20 events in memory.
        """
        hora = datetime.datetime.now().strftime("%H:%M:%S")
        mensaje = f"{hora} - {objeto} - Zona {zona_idx+1}"
        self.events.append(mensaje)
        print(mensaje)
        self._save_to_txt(mensaje)
        if frame_final is not None:
            self._save_capture(frame_final, objeto, zona_idx)
        if len(self.events) > 20:
            self.events.pop(0)

    def _save_to_txt(self, mensaje):
        """
        Saves a log message to the text file.

        Parameters
        ----------
        mensaje : str
            The log message to write.

        Notes
        -----
        - Appends the message to `eventos.txt` in `LOGS_DIR`.
        - Creates the logs directory if it doesn't exist.
        """
        os.makedirs(LOGS_DIR, exist_ok=True)
        with open(os.path.join(LOGS_DIR, "eventos.txt"), "a", encoding="utf-8") as f:
            f.write(mensaje + "\n")

    def _save_capture(self, frame, objeto, zona_idx):
        """
        Saves an image capture of the event.

        Parameters
        ----------
        frame : numpy.ndarray
            The frame image to save.
        objeto : str
            The label of the detected object.
        zona_idx : int
            Index of the restricted zone (0-based).

        Notes
        -----
        - Creates the captures directory if it doesn't exist.
        - Filenames follow the format:
          `captura_<objeto>_zona<zone_number>_<YYYY-MM-DD_HH-MM-SS>.png`
        """

        os.makedirs(CAPTURES_DIR, exist_ok=True)
        hora_archivo = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        nombre = f"captura_{objeto}_zona{zona_idx+1}_{hora_archivo}.png"
        cv2.imwrite(os.path.join(CAPTURES_DIR, nombre), frame)

    def draw_on_frame(self, frame):
        """
        Draws the latest event messages on the provided frame.

        Parameters
        ----------
        frame : numpy.ndarray
            The image frame where event messages will be overlaid.

        Returns
        -------
        numpy.ndarray
            The frame with the event log overlay.

        Notes
        -----
        - Draws up to `MAX_LOGS_DISPLAY` events in reverse chronological order.
        - Event text is drawn with a shadow effect for visibility.
        """
        if not self.events:
            return frame
        h, w = frame.shape[:2]
        x0 = w - 320
        y0 = h - LOG_AREA_OFFSET_Y
        for i, msg in enumerate(reversed(self.events[-MAX_LOGS_DISPLAY:])):
            y = y0 + i * 20
            cv2.putText(frame, msg, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 2)
            cv2.putText(frame, msg, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame
