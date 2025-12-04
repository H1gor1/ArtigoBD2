import os
import subprocess


class Downloader:
    def __init__(self):
        self.path = os.path.join(os.getcwd(), "musicas_teste")

    def download(self, link):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        subprocess.run(
            [
                "yt-dlp",
                "-f",
                "bestaudio",
                "--extract-audio",
                "--audio-format",
                "mp3",
                link,
            ],
            cwd=self.path,
        )

        pass
