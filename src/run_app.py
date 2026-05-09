import sys
from pathlib import Path
import streamlit.web.bootstrap as stb

if __name__ == "__main__":
    app_path = Path(__file__).with_name("myapp.py")
    sys.argv = ["streamlit", "run", str(app_path)]
    stb.run()