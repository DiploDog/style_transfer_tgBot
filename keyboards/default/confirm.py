from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

confirm = ReplyKeyboardMarkup(
    keyboard=[
    [
        KeyboardButton(text='Отмена'),
        KeyboardButton(text='Поехали!')
    ],
  ],
    resize_keyboard=True
)