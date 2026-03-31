"""
Audio level visualizer — runs as a separate process via multiprocessing.
Must live in its own module so PyInstaller's spawn-based multiprocessing
can import it by name rather than looking it up in __main__.
"""

def run_visualizer_process():
    import tkinter as tk
    import numpy as np
    import sounddevice as sd

    root = tk.Tk()
    root.title("Kura Pegel")
    root.geometry("200x50+50+50")
    root.attributes("-topmost", True)
    root.overrideredirect(True)
    canvas = tk.Canvas(root, width=200, height=50, bg='black', highlightthickness=0)
    canvas.pack()

    def update_wave():
        try:
            recording = sd.rec(int(0.1 * 44100), samplerate=44100, channels=1)
            sd.wait()
            volume = np.linalg.norm(recording) * 10
            canvas.delete("all")
            width = min(volume * 200, 200)
            color = "#00FF00" if width < 160 else "#FF3B30"
            canvas.create_rectangle(0, 0, width, 50, fill=color, outline="")
            root.after(50, update_wave)
        except Exception:
            root.destroy()

    update_wave()
    root.mainloop()