
def progress_print(p, _last_pstr=['']):
    assert 0 <= p <= 1
    pstr = '{:.2f}'.format(p)
    if _last_pstr[0] != pstr:
        _last_pstr[0] = pstr
        print('\r{}'.format(pstr), end='')

