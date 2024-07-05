from jinja2 import Environment, FileSystemLoader, Template
from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext
import telebot

file_loader = FileSystemLoader("templates")
env = Environment(loader=file_loader)

bot = telebot.TeleBot("token")



welcome_template = env.get_template("welcome_template.txt")

#user_name = "Саня"
bot_name = "Большой Биг"

@bot.message_handler(commands=["start"])
def main(chat):
    first_welcome_message = welcome_template.render(user_name=chat.from_user.first_name, bot_name=bot_name)
    bot.send_message(chat.chat.id, first_welcome_message, parse_mode="html")

bot.infinity_polling()