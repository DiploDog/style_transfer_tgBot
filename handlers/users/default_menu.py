from aiogram.utils.emoji import emojize
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.dispatcher.filters import Command
from keyboards.inline.credits_buttons import credits_buttons
from loader import dp, bot
from aiogram.types import Message, ReplyKeyboardRemove, ContentType, InputFile
from keyboards.default.menu import menu
from keyboards.default.confirm import confirm
from model import run_transfer
import os


class Form(StatesGroup):
    style = State()
    content = State()
    transfer = State()


@dp.message_handler(Command('menu'))
async def show_menu(message: Message):
    await message.answer(
        'Нажмите любую кнопку',
        reply_markup=menu
    )


@dp.message_handler(commands=['сюда'], state='*')
async def error_message(message: Message):
    if not os.path.isfile('images/result.jpg'):
        await message.answer(
            'Боту не хватило ресурсов.\n'
            'Попробуйте еще раз...'
        )


@dp.message_handler(text='Поработаем со стилем!', state='*')
async def start_transfer(message: Message):
    await message.answer(
        'Отлично! Пришлите мне по очереди две фотографии:\n'
        '1) фото, стиль которого Вы хотели бы использовать\n'
        '2) фото, для которого предназначаются изменения стиля\n\n'
        'Чтобы начать, отправьте мне фото, стиль которого вы хотели бы использовать.',
        reply_markup=ReplyKeyboardRemove()
    )
    await Form.style.set()


@dp.message_handler(content_types=['photo'], state=Form.style)
async def get_style(message: ContentType.PHOTO, state: FSMContext):
    await state.get_data()
    file_id = message.photo[-1].file_id
    await state.update_data(style_id=file_id)
    await message.answer('Отлично! Теперь фото, для которого предназначается изменение стиля')
    await Form.content.set()


@dp.message_handler(content_types=['photo'], state=Form.content)
async def get_content(message: ContentType.PHOTO, state: FSMContext):
    await state.get_data()
    file_id = message.photo[-1].file_id
    await state.update_data(content_id=file_id)
    await message.answer(f'Получил второе фото!\nВсё готово для обработки\n'
                         f'Ваших фото, {message.from_user.full_name}!\nЧтобы продолжить, '
                         f'нажмите "Поехали".',
                         reply_markup=confirm)
    await Form.transfer.set()


@dp.message_handler(text='Поехали!', state=Form.transfer)
async def process_transfer(message: Message, state: FSMContext):
    data = await state.get_data()
    content = data.get('content_id')
    style = data.get('style_id')
    await bot.download_file_by_id(content, 'images/content.jpg')
    await bot.download_file_by_id(style, 'images/style.jpg')
    await message.answer(
        emojize('Обработка Вашего фото началсь!\n'
                'Обычно это занимает от 10 до 15 минут.\n'
                'Пожалуйста, наберитесь терпения :point_right::point_left:\n'
                'Нажмите /сюда и в случае, если случится сбой\n'
                'я Вас обязательно об этом оповещу!'),
        reply_markup=ReplyKeyboardRemove()
    )
    if os.path.isfile('images/result.jpg'):
        os.remove('images/result.jpg')
    run_transfer.run('images/content.jpg', 'images/style.jpg')
    await bot.send_photo(message.chat.id, InputFile('images/result.jpg'), caption='Готово!')
    await state.finish()


@dp.message_handler(text='Отмена', state=Form.transfer)
async def cancel_transfer(message: Message, state: FSMContext):
    await state.finish()
    await message.answer(
        'Процесс переноса стиля отменен!',
        reply_markup=menu
    )


# @dp.message_handler(text='Режим разработчика')
# async def get_csv_logs(message: Message):
#     # TODO: Прислать файл .csv c последними логами лоссов
#     await message.answer(
#         f'Вы нажали {message.text}. Спасибо!',
#         reply_markup=ReplyKeyboardRemove()
#     )


@dp.message_handler(text='Об авторе')
async def get_credits(message: Message):
    await message.answer(
        'Спасибо за проявляенный интерес к автору!\n'
        'Ссылки для связи:\n',
        reply_markup=credits_buttons
    )


