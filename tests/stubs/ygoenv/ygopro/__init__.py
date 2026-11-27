import ygoenv


def init_module(db_path, code_list_file, decks):
    ygoenv.REGISTERED.update(decks)
