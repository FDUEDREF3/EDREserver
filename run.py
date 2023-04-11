from app.config.settings import Config
from app.create_app import create_app

app = create_app(Config)

if __name__ == "__main__":
    app.run(host='0.0.0.0')
