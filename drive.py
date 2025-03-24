import gdown

file_id = "1DpNn2O6horKfS6sASTzfgliObG-5AqfC"
url = f"https://drive.google.com/file/d/1DpNn2O6horKfS6sASTzfgliObG-5AqfC/view?usp=drive_link"
output = "checkpoint.pth"

gdown.download(url, output, quiet=False)