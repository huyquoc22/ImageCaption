
class Config(object):
    def __init__(self, default, overwrite):
        self.default = default
        self.overwrite = overwrite

    def get(self, k):
        if dict_has_path(self.default, k):
            base = dict_get_path_value(self.default, k)
        else:
            base = None
        if dict_has_path(self.overwrite, k):
            over = dict_get_path_value(self.overwrite, k)
            if isinstance(base, dict):
                assert isinstance(over, dict)
                base.update(over)
            else:
                base = over
        return base

    def __getattr__(self, k):
        return self.get(k)

    def __copy__(self):
        return Config(self.default, self.overwrite)

    def __deepcopy__(self, memo):
        from copy import deepcopy
        return Config(deepcopy(self.default), deepcopy(self.overwrite))

    def get_dict(self):
        import copy
        default = copy.deepcopy(self.default)
        for p in get_all_path(self.overwrite, with_list=False):
            v = dict_get_path_value(self.overwrite, p)
            dict_update_path_value(default, p, v)
        return default

def dict_remove_path(d, p):
    ps = p.split('$')
    assert len(ps) > 0
    cur_dict = d
    need_delete = ()
    while True:
        if len(ps) == 1:
            if len(need_delete) > 0 and len(cur_dict) == 1:
                del need_delete[0][need_delete[1]]
            else:
                del cur_dict[ps[0]]
            return
        else:
            if len(cur_dict) == 1:
                if len(need_delete) == 0:
                    need_delete = (cur_dict, ps[0])
            else:
                need_delete = (cur_dict, ps[0])
            cur_dict = cur_dict[ps[0]]
            ps = ps[1:]

def dict_has_path(d, p, with_type=False):
    ps = p.split('$')
    cur_dict = d
    while True:
        if len(ps) > 0:
            k = dict_parse_key(ps[0], with_type)
            if isinstance(cur_dict, dict) and k in cur_dict:
                cur_dict = cur_dict[k]
                ps = ps[1:]
            elif isinstance(cur_dict, list):
                try:
                    k = int(k)
                except:
                    return False
                cur_dict = cur_dict[k]
                ps = ps[1:]
            else:
                return False
        else:
            return True

def dict_update_path_value(d, p, v):
    ps = p.split('$')
    while True:
        if len(ps) == 1:
            d[ps[0]] = v
            break
        else:
            if ps[0] not in d:
                d[ps[0]] = {}
            d = d[ps[0]]
            ps = ps[1:]

def dict_parse_key(k, with_type):
    if with_type:
        if k[0] == 'i':
            return int(k[1:])
        else:
            return k[1:]
    return k

def dict_get_path_value(d, p, with_type=False):
    ps = p.split('$')
    cur_dict = d
    while True:
        if len(ps) > 0:
            k = dict_parse_key(ps[0], with_type)
            if isinstance(cur_dict, (tuple, list)):
                cur_dict = cur_dict[int(k)]
            else:
                cur_dict = cur_dict[k]
            ps = ps[1:]
        else:
            return cur_dict


def get_all_path(d, with_type=False, leaf_only=True, with_list=True):
    assert not with_type, 'will not support'
    all_path = []

    if isinstance(d, dict):
        for k, v in d.items():
            all_sub_path = get_all_path(
                v, with_type, leaf_only=leaf_only, with_list=with_list)
            all_path.extend([k + '$' + p for p in all_sub_path])
            if not leaf_only or len(all_sub_path) == 0:
                all_path.append(k)
    elif (isinstance(d, tuple) or isinstance(d, list)) and with_list:
        for i, _v in enumerate(d):
            all_sub_path = get_all_path(
                _v, with_type,
                leaf_only=leaf_only,
                with_list=with_list,
            )
            all_path.extend(['{}$'.format(i) + p for p in all_sub_path])
            if not leaf_only or len(all_sub_path) == 0:
                all_path.append('{}'.format(i))
    return all_path


