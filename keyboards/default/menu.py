from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

menu = ReplyKeyboardMarkup(
    keyboard=[
    [
        KeyboardButton(text='Поработаем со стилем!'),
    ],
    [
        KeyboardButton(text='Об авторе'),
    ],
  ],
    resize_keyboard=True
)