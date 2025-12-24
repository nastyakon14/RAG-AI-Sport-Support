from aiogram.utils.keyboard import InlineKeyboardBuilder


def feedback_kb(response_id: int):
    kb = InlineKeyboardBuilder()
    kb.button(text="👍", callback_data=f"fb:1:{response_id}")
    kb.button(text="👎", callback_data=f"fb:0:{response_id}")
    kb.adjust(2)

    return kb.as_markup()
