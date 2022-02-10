from environs import Env

env = Env()
env.read_env()

BOT_TOKEN = env.str("BOT_TOKEN")
ADMINS = env.list("ADMINS")
WEBHOOK_HOST = env.str("WEBHOOK_HOST")
WEBHOOK_PATH = env.str("WEBHOOK_PATH")
WEBHOOK_URL = env.str("WEBHOOK_URL")
WEBAPP_HOST = env.str("WEBAPP_HOST")
WEBAPP_PORT = 3001
