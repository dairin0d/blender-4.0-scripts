# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

import sys
import itertools
import inspect
from collections import namedtuple

import bpy

from mathutils import Vector, Matrix, Quaternion, Euler, Color

from idprop.types import IDPropertyArray, IDPropertyGroup

import numpy as np

from .utils_python import issubclass_safe
from .utils_math import matrix_flatten, matrix_unflatten
from .utils_text import split_camelcase, compress_whitespace

#============================================================================#

# See also Blender's rna_info module

bpy_struct = bpy.types.AnyType.__base__

class BlRna:
    rna_to_bpy = {
        "BoolProperty":(bpy.props.BoolProperty, bpy.props.BoolVectorProperty),
        "IntProperty":(bpy.props.IntProperty, bpy.props.IntVectorProperty),
        "FloatProperty":(bpy.props.FloatProperty, bpy.props.FloatVectorProperty),
        "StringProperty":bpy.props.StringProperty,
        "EnumProperty":bpy.props.EnumProperty,
        "PointerProperty":bpy.props.PointerProperty,
        "CollectionProperty":bpy.props.CollectionProperty,
    }
    
    math_types = {
        Color: tuple,
        Euler: tuple,
        Quaternion: tuple,
        Vector: tuple,
        Matrix: (lambda m: tuple(tuple(v) for v in m)),
    }
    
    def __new__(cls, obj):
        # Note: operator instances have both rna_type and bl_rna,
        # but actual user-defined properties are present only in rna_type
        
        try:
            return obj.rna_type # e.g. operator instances
        except (AttributeError, TypeError, KeyError):
            pass
        
        try:
            return obj.bl_rna # e.g. descendants of bpy_struct
        except AttributeError:
            pass
        
        try:
            return obj.get_rna().bl_rna
        except (AttributeError, TypeError, KeyError):
            pass
        
        try:
            return obj.get_rna_type() # e.g. bpy.ops operators
        except (AttributeError, TypeError, KeyError):
            pass
        
        return None
    
    @staticmethod
    def is_valid(obj):
        if not obj: return False
        try:
            # TypeError: descriptor 'as_pointer' of 'bpy_struct' object needs an argument
            #obj.as_pointer()
            id_data = obj.id_data
            return True
        except (ReferenceError, AttributeError):
            return False
    
    @staticmethod
    def iter_valid(iterable):
        return (item for item in iterable if BlRna.is_valid(item))
    
    @staticmethod
    def parent(obj, n=-1, coerce=True):
        path_parts = obj.path_from_id().split(".")
        if len(path_parts) <= 1: return obj.id_data
        return obj.id_data.path_resolve(".".join(path_parts[:n]), coerce)
    
    @staticmethod
    def full_path(obj):
        if isinstance(obj, bpy.types.ID): return repr(obj.id_data)
        return repr(obj.id_data) + "." + obj.path_from_id()
    
    @staticmethod
    def full_path_resolve(full_path, coerce=True):
        return eval(full_path) # for now
    
    @staticmethod
    def enum_to_int(obj, name, value=None):
        if value is None: value = getattr(obj, name)
        rna_prop = BlRna(obj).properties[name]
        if rna_prop.is_enum_flag:
            res = 0
            for item in rna_prop.enum_items:
                if item.identifier in value: res |= item.value
            return res
        else:
            for item in rna_prop.enum_items:
                if item.identifier == value: return item.value
            return 0
    
    @staticmethod
    def enum_from_int(obj, name, value):
        rna_prop = BlRna(obj).properties[name]
        if rna_prop.is_enum_flag:
            return {item.identifier for item in rna_prop.enum_items if item.value & value}
        else:
            for item in rna_prop.enum_items:
                if item.value == value: return item.identifier
            return None
    
    @staticmethod
    def enum_items(obj, name, container=list):
        if isinstance(obj, str): obj = getattr(bpy.types, obj)
        
        rna_prop = BlRna(obj).properties[name]
        enum_items = rna_prop.enum_items
        
        if not enum_items:
            # RNA enum items may be empty also when dynamic items are used.
            # If obj is based on a python class, we can try checking this explicitly.
            function, keywords = BpyProp.find(obj, name, unwrap=True)
            if function:
                items = keywords.get("items")
                if callable(items):
                    try:
                        enum_items = items(obj, bpy.context)
                    except Exception:
                        pass
        
        return BpyEnum.normalize_items(enum_items, rna_prop.is_enum_flag, separators=False, container=container, wrap=True)
    
    @staticmethod
    def to_bpy_prop(obj, name=None):
        rna_prop = (obj if name is None else BlRna(obj).properties[name])
        
        type_id = rna_prop.bl_rna.identifier
        function = BlRna.rna_to_bpy.get(type_id)
        if not function: return None
        
        # Note: DON'T assign tags, they are not available for all classes
        keywords = dict(name=rna_prop.name, description=rna_prop.description,
            translation_context=rna_prop.translation_context,
            options=set(), override=set())
        
        def map_option(rna_name, bpy_name):
            if getattr(rna_prop, rna_name, False):
                keywords["options"].add(bpy_name)
        
        def map_override(rna_name, bpy_name):
            if getattr(rna_prop, rna_name, False):
                keywords["override"].add(bpy_name)
        
        def map_keyword(rna_name, bpy_name, is_vector_size=False):
            # Some built-in enum properties may have incorrectly specified RNA, leading to
            # warnings like "pyrna_enum_to_py: current value ... matches no enum in ..."
            # The best we can do is merely to minimize the number of such warnings.
            value = getattr(rna_prop, rna_name, BlRna) # any invalid default will do
            if value is not BlRna:
                if is_vector_size:
                    value = tuple(v for v in value if v > 0)
                    keywords[bpy_name] = (value if len(value) > 1 else value[0])
                else:
                    keywords[bpy_name] = BlRna.serialize_value(value, False)
        
        map_option("is_hidden", 'HIDDEN')
        map_option("is_skip_save", 'SKIP_SAVE')
        map_option("is_animatable", 'ANIMATABLE')
        map_option("is_library_editable", 'LIBRARY_EDITABLE')
        #map_option("", 'PROPORTIONAL') # no correspondence?
        if type_id == "EnumProperty":
            map_option("is_enum_flag", 'ENUM_FLAG')
        
        map_override("is_overridable", 'LIBRARY_OVERRIDABLE')
        
        if hasattr(rna_prop, "array_length"):
            if rna_prop.array_length == 0:
                function = function[0] # bool/int/float
                map_keyword("default", "default")
            else:
                function = function[1] # bool/int/float vector
                map_keyword("array_dimensions", "size", True)
                map_keyword("default_array", "default")
            map_keyword("subtype", "subtype")
            map_keyword("hard_min", "min")
            map_keyword("soft_min", "soft_min")
            map_keyword("hard_max", "max")
            map_keyword("soft_max", "soft_max")
            map_keyword("step", "step")
            map_keyword("precision", "precision")
        elif type_id == "StringProperty":
            map_keyword("default", "default")
            map_keyword("subtype", "subtype")
            map_keyword("length_max", "maxlen")
        elif type_id == "EnumProperty":
            map_keyword(("default_flag" if rna_prop.is_enum_flag else "default"), "default")
            map_keyword("enum_items", "items")
            keywords["default"] = BpyEnum.fix_value(keywords["items"], keywords["default"])
        else:
            # Caution: fixed_type returns a blender struct, not the original class
            map_keyword("fixed_type", "type")
        
        return BpyProp.wrap(function, keywords)
    
    @staticmethod
    def properties(obj):
        # first rna property item is always rna_type (?)
        return BlRna(obj).properties.items()[1:]
    
    @staticmethod
    def functions(obj):
        return BlRna(obj).functions[funcname].items()
    
    @staticmethod
    def parameters(obj, funcname):
        return BlRna(obj).functions[funcname].parameters.items()
    
    # NOTE: we can't just compare value to the result of get_default(),
    # because in some places Blender's return values not always correspond
    # to the subtype declared in rna (e.g. TRANSFORM_OT_translate.value
    # returns a Vector, but its subtype is 'NONE')
    @staticmethod
    def is_default(value, obj, name=None):
        rna_prop = (obj if name is None else BlRna(obj).properties[name])
        type_id = rna_prop.bl_rna.identifier
        if hasattr(rna_prop, "array_length"):
            if rna_prop.array_length == 0: return value == rna_prop.default
            if isinstance(value, Matrix): value = itertools.chain(*value.col)
            return tuple(rna_prop.default_array) == tuple(value)
        elif type_id == "StringProperty":
            return value == rna_prop.default
        elif type_id == "EnumProperty":
            if rna_prop.is_enum_flag: return set(value) == rna_prop.default_flag
            return value == rna_prop.default
        return False
    
    @staticmethod
    def get_default(obj, name=None):
        rna_prop = (obj if name is None else BlRna(obj).properties[name])
        type_id = rna_prop.bl_rna.identifier
        if hasattr(rna_prop, "array_length"):
            if rna_prop.array_length == 0: return rna_prop.default
            type_id = rna_prop.bl_rna.identifier
            if type_id != "FloatProperty": return tuple(rna_prop.default_array)
            return BlRna._convert_float_array(rna_prop)
        elif type_id == "StringProperty":
            return rna_prop.default
        elif type_id == "EnumProperty":
            return (rna_prop.default_flag if rna_prop.is_enum_flag else rna_prop.default)
        return None
    
    def _convert_float_array():
        def convert_color(rna_prop):
            return Color(rna_prop.default_array)
        def convert_vector(rna_prop):
            return Vector(rna_prop.default_array)
        def convert_matrix(rna_prop):
            # For now, flatten before unflattening (since array may potentially be multidimensional)
            return matrix_unflatten(np.ravel(rna_prop.default_array, order='C'))
        def convert_euler_quaternion(rna_prop):
            return (Euler if rna_prop.array_length == 3 else Quaternion)(rna_prop.default_array)
        
        color = ((3,), convert_color)
        vector = ((2,3,4), convert_vector)
        matrix = ((9,16), convert_matrix)
        euler_quaternion = ((3,4), convert_euler_quaternion)
        
        math_types = {
            'COLOR':color, 'COLOR_GAMMA':color,
            'TRANSLATION':vector, 'DIRECTION':vector, 'VELOCITY':vector, 'ACCELERATION':vector, 'XYZ':vector,
            'MATRIX':matrix,
            'EULER':euler_quaternion, 'QUATERNION':euler_quaternion,
        }
        
        @staticmethod
        def convert(rna_prop):
            math_type = math_types.get(rna_prop.subtype)
            if (math_type is None) or (rna_prop.array_length not in math_type[0]):
                return tuple(rna_prop.default_array)
            else:
                return math_type[1](rna_prop)
        
        return convert
    
    _convert_float_array = _convert_float_array()
    
    @staticmethod
    def reset(obj, ignore_default=False):
        if hasattr(obj, "property_unset"): # method of bpy_struct
            for name, rna_prop in BlRna.properties(obj):
                obj.property_unset(name)
        else:
            for name, rna_prop in BlRna.properties(obj):
                if (not ignore_default) or (not BlRna.is_default(getattr(obj, name), rna_prop)):
                    setattr(obj, name, BlRna.get_default(rna_prop))
    
    @staticmethod
    def serialize_value(value, recursive=True, json=False, structs=False, is_struct=False):
        """Serialize rna property value"""
        if isinstance(value, bpy.types.ID):
            # We have to store ID block type along with its name,
            # since some properties don't reference specific
            # ID types (e.g. Object.data's type is ID)
            value = (type(value).__name__, value.name_full)
        else:
            value_class = value.__class__
            base_class = value_class.__base__
            class_name = value_class.__name__
            
            if (base_class is bpy.types.PropertyGroup) or (structs and is_struct):
                if recursive:
                    rna_names = ("bl_rna", "rna_type")
                    value = {
                        rna_prop.identifier:BlRna.serialize_value(getattr(value, rna_prop.identifier),
                            recursive=recursive, structs=structs, is_struct=rna_prop.is_never_none)
                        for name, rna_prop in BlRna(value).properties.items() if name not in rna_names
                    }
            elif value_class is bpy.types.EnumPropertyItem:
                value = (value.identifier, value.name, value.description, value.icon, value.value)
            elif class_name == "bpy_prop_array":
                value = tuple(value) # bool/int/float vector
            elif class_name == "bpy_prop_collection":
                value = [BlRna.serialize_value(item, recursive) for item in value]
            elif class_name == "bpy_prop_collection_idprop":
                value = [BlRna.serialize_value(item, recursive) for item in value]
            elif value_class in BlRna.math_types:
                value = (BlRna.math_types[value_class](value) if json else value.copy().freeze())
        
        if json:
            if isinstance(value, set):
                value = list(value) # json.dumps() does not accept sets
        
        return value
    
    @staticmethod
    def serialize(obj, ignore_default=False, json=False, structs=False):
        """Serialize object's rna properties"""
        if not obj: return None
        data = {}
        for name, rna_prop in BlRna.properties(obj):
            name = rna_prop.identifier # should always be identical, but just to be sure
            if ignore_default and (not obj.is_property_set(name)): continue
            value = getattr(obj, name)
            data[name] = BlRna.serialize_value(value, json=json,
                structs=structs, is_struct=rna_prop.is_never_none)
        return data
    
    @staticmethod
    def deserialize(obj, data, ignore_default=False, suppress_errors=False):
        """Deserialize object's rna properties"""
        if (not obj) or (not data): return
        
        if not isinstance(data, dict): data = BlRna.serialize(data)
        
        rna_props = BlRna(obj).properties
        for name, value in data.items():
            rna_prop = rna_props.get(name)
            if rna_prop is None: continue
            
            type_id = rna_prop.bl_rna.identifier
            
            if type_id == "PointerProperty":
                if isinstance(value, bpy_struct) and not (rna_prop.is_readonly or rna_prop.is_never_none):
                    try:
                        setattr(obj, name, value)
                    except:
                        # sometimes Blender's rna is incomplete/incorrect
                        if not suppress_errors: raise
                elif isinstance(rna_prop.fixed_type, bpy.types.ID):
                    if rna_prop.is_readonly: continue
                    
                    if value is not None:
                        bpy_data = BpyData.to_data(value[0])
                        value = bpy_data.get(value[1])
                    
                    try:
                        setattr(obj, name, value)
                    except:
                        # sometimes Blender's rna is incomplete/incorrect
                        if not suppress_errors: raise
                else:
                    BlRna.deserialize(getattr(obj, name), value, ignore_default, suppress_errors)
            elif type_id == "CollectionProperty":
                collection = getattr(obj, name)
                # collection.add() generally only works for bpy.props.CollectionProperty
                if collection.__class__.__name__ != "bpy_prop_collection_idprop": continue
                
                collection.clear()
                if isinstance(rna_prop.fixed_type, bpy.types.ID):
                    # Blender does not yet support defining ID collection properties; this is just in case
                    for item in value:
                        collection.add()
                        if item is not None:
                            bpy_data = BpyData.to_data(item[0])
                            collection[-1] = bpy_data.get(item[1])
                else:
                    for item in value:
                        BlRna.deserialize(collection.add(), item, ignore_default, suppress_errors)
            else:
                if rna_prop.is_readonly: continue
                
                if (not ignore_default) or (not BlRna.is_default(value, rna_prop)):
                    if (type_id == "EnumProperty") and rna_prop.is_enum_flag and (not isinstance(value, set)):
                        value = set(value) # might be other collection type when loaded from JSON
                    
                    try:
                        setattr(obj, name, value)
                    except:
                        # sometimes Blender's rna is incomplete/incorrect
                        if not suppress_errors: raise
    
    @staticmethod
    def compare_prop(rna_prop, valueA, valueB):
        if rna_prop.type == 'POINTER':
            ft = rna_prop.fixed_type
            if isinstance(ft, bpy.types.ID):
                # idblocks are used only by reference
                return valueA == valueB
            return BlRna.compare(valueA, valueB)
        elif rna_prop.type == 'COLLECTION':
            if len(valueA) != len(valueB): return False
            return all(BlRna.compare(valueA[i], valueB[i]) for i in range(len(valueA)))
        else: # primitive types or enum
            if hasattr(rna_prop, "array_length") and not isinstance(valueA, (Matrix, str)):
                if (rna_prop.array_length != 0) or hasattr(valueA, "__len__"):
                    return (len(valueA) == len(valueB)) and (tuple(valueA) == tuple(valueB))
            return valueA == valueB
    
    @staticmethod
    def compare(objA, objB, ignore=(), specials={}):
        """Compare objects' rna properties"""
        if (objA is None) and (objB is None): return True
        if (objA is None) or (objB is None): return False
        if objA == objB: return True
        # objects are expected to be of the same type
        for name, rna_prop in BlRna.properties(objA):
            if name in ignore: continue
            valueA = getattr(objA, name)
            valueB = getattr(objB, name)
            if name in specials:
                if not specials[name](rna_prop, valueA, valueB): return False
            elif not BlRna.compare_prop(rna_prop, valueA, valueB):
                #print(f"Not same: {name} in {type(valueA)}/{type(valueB)}")
                return False
        return True

class BpyOp:
    @staticmethod
    def convert(idname, py):
        op_parts = (idname.split(".") if "." in idname else idname.rsplit("_OT_", 1))
        if py: return f"{op_parts[-2].lower()}.{op_parts[-1].lower()}"
        return f"{op_parts[-2].upper()}_OT_{op_parts[-1].lower()}"
    
    def __new__(cls, op):
        if isinstance(op, BpyOp): return op # just in case
        
        try:
            if isinstance(op, str):
                op_parts = (op.split(".") if "." in op else op.rsplit("_OT_", 1))
                category = getattr(bpy.ops, op_parts[-2].lower())
                op = getattr(category, op_parts[-1].lower())
            rna = op.get_rna_type()
            if not rna: return None
        except (AttributeError, KeyError, IndexError):
            return None
        
        self = object.__new__(cls)
        self.op = op
        self.rna = rna
        return self
    
    def __call__(self, *args, **kwargs):
        self.op(*args, **kwargs)
    
    # Emulate operator's public API
    get_rna_type = property(lambda self: self.op.get_rna_type)
    idname = property(lambda self: self.op.idname)
    idname_py = property(lambda self: self.op.idname_py)
    poll = property(lambda self: self.op.poll)
    
    type = property(lambda self: bpy.types.Operator.bl_rna_get_subclass_py(self.op.idname()))

class BpyProp:
    """Utility class for easy inspection/modification of bpy properties"""
    
    base_type = type(bpy.props.BoolProperty) # builtin_function_or_method
    
    wrapper_type = bpy.props._PropertyDeferred
    wrapper_types = (wrapper_type,)
    
    unwrapper_type = namedtuple("BpyPropUnwrapped", ["function", "keywords"])
    
    def _make_defaults(has_update=True, has_get_set=True, **kwargs):
        result = dict(name="", description="", options={'ANIMATABLE'}, tags=set(), **kwargs)
        result.update(override=set())
        if has_update: result.update(update=None)
        if has_get_set: result.update(get=None, set=None)
        return result
    
    # I have no idea how to get the default values using reflection
    # (it seems like bpy.props.* functions have no python-accessible signature)
    known = {
        bpy.props.BoolProperty:_make_defaults(
            default=False, subtype='NONE',
        ),
        bpy.props.BoolVectorProperty:_make_defaults(
            default=(False, False, False), size=3, subtype='NONE',
        ),
        bpy.props.IntProperty:_make_defaults(
            default=0, step=1, subtype='NONE',
            min=-2**31, soft_min=-2**31,
            max=2**31-1, soft_max=2**31-1,
        ),
        bpy.props.IntVectorProperty:_make_defaults(
            default=(0, 0, 0), size=3, step=1, subtype='NONE',
            min=-2**31, soft_min=-2**31,
            max=2**31-1, soft_max=2**31-1,
        ),
        bpy.props.FloatProperty:_make_defaults(
            default=0.0, step=3, precision=2, subtype='NONE', unit='NONE',
            min=sys.float_info.min, soft_min=sys.float_info.min,
            max=sys.float_info.max, soft_max=sys.float_info.max,
        ),
        bpy.props.FloatVectorProperty:_make_defaults(
            default=(0.0, 0.0, 0.0), size=3, step=3, precision=2, subtype='NONE', unit='NONE',
            min=sys.float_info.min, soft_min=sys.float_info.min,
            max=sys.float_info.max, soft_max=sys.float_info.max,
        ),
        bpy.props.StringProperty:_make_defaults(
            default="", subtype='NONE', maxlen=0,
        ),
        bpy.props.EnumProperty:_make_defaults(
            default=None, items=None,
        ),
        bpy.props.PointerProperty:_make_defaults(True, False,
            poll=None, type=None,
        ),
        bpy.props.CollectionProperty:_make_defaults(False, False,
            type=None,
        ),
    }
    
    vectors = (bpy.props.BoolVectorProperty,
               bpy.props.IntVectorProperty,
               bpy.props.FloatVectorProperty)
    
    @staticmethod
    def wrap(function, keywords):
        return function(**keywords)
    
    @staticmethod
    def unwrap(bpy_prop, exact=False):
        if isinstance(bpy_prop, (BpyProp.wrapper_types if exact else BpyProp.wrapper_types_extended)):
            return BpyProp.unwrapper_type(bpy_prop.function, bpy_prop.keywords)
        elif isinstance(bpy_prop, tuple) and (len(bpy_prop) == 2) and BpyProp.is_wrappable(*bpy_prop):
            return BpyProp.unwrapper_type(*bpy_prop)
        return BpyProp.unwrapper_type(None, None)
    
    @staticmethod
    def is_wrappable(function, keywords):
        # Before checking presence in BpyProp.known, make sure it's a hashable type
        if not isinstance(function, BpyProp.base_type): return False
        return (function in BpyProp.known) and isinstance(keywords, dict)
    
    @staticmethod
    def validate(value):
        """Test whether a given object is a bpy property"""
        return isinstance(value, BpyProp.wrapper_type)
    
    @staticmethod
    def find(cls, name, unwrap=False):
        if not isinstance(cls, type): cls = type(cls)
        
        annotations = getattr(cls, "__annotations__", None)
        bpy_prop = (annotations.get(name) if annotations else None)
        
        if unwrap:
            unwrapped = BpyProp.unwrap(bpy_prop, True)
            if unwrapped.function: return unwrapped
            
            bpy_prop = getattr(cls, name, None)
            return BpyProp.unwrap(bpy_prop, True)
        else:
            if BpyProp.validate(bpy_prop): return bpy_prop
            
            bpy_prop = getattr(cls, name, None)
            return (bpy_prop if BpyProp.validate(bpy_prop) else None)
    
    @staticmethod
    def iterate(cls, exclude_hidden=False, names=None, static=True, dynamic=True, inherited=False):
        """Iterate over bpy properties in a class"""
        if not isinstance(cls, type): cls = type(cls)
        if isinstance(names, str): names = (names,)
        
        # Note: python classes seem to always have __dict__
        # (a mappingproxy object); __slots__ affect only instances
        def _iterate(cls, storage):
            storage = getattr(cls, storage, None)
            if not storage: return
            for name in (storage.keys() if names is None else names):
                if name.startswith("_"): continue # bpy prop name cannot start with an underscore
                prop_info = BpyProp(storage.get(name))
                if not prop_info: continue
                if exclude_hidden and ('HIDDEN' in prop_info.get("options")): continue
                yield name, prop_info
        
        # It seems that in some cases Blender ignores annotations/attributes in parent classes?
        
        if inherited:
            def _iterate_inherited(cls):
                for base in cls.__bases__:
                    if base is object: continue
                    yield from _iterate_inherited(base)
                    if static: yield from _iterate(base, "__annotations__")
                    if dynamic: yield from _iterate(base, "__dict__")
            yield from _iterate_inherited(cls)
        
        if static: yield from _iterate(cls, "__annotations__")
        if dynamic: yield from _iterate(cls, "__dict__")
    
    @staticmethod
    def is_in(cls, exclude_hidden=False):
        """Test whether a given class contains any bpy properties"""
        return any(BpyProp.iterate(cls, exclude_hidden=exclude_hidden))
    
    @staticmethod
    def reset(obj, names=None, recursive=False):
        """Reset properties of an instance to their default values"""
        for key, info in BpyProp.iterate(type(obj), names=names):
            if info.function == bpy.props.PointerProperty:
                if recursive:
                    _obj = getattr(obj, key)
                    BpyProp.reset(_obj, None, True)
            elif info.function == bpy.props.CollectionProperty:
                if recursive:
                    for _obj in getattr(obj, key):
                        BpyProp.reset(_obj, None, True)
            else:
                if hasattr(obj, "property_unset"):
                    obj.property_unset(key)
                else:
                    setattr(obj, key, info["default"])
    
    @staticmethod
    def deserialize(obj, data, cls=None, names=None, use_skip_save=False):
        """Deserialize object properties from a JSON data structure"""
        
        if not cls: cls = type(obj)
        
        for key, info in BpyProp.iterate(cls, names=names):
            if use_skip_save and ('SKIP_SAVE' in info.get("options", "")): continue
            
            try:
                data_value = data[key]
            except TypeError:
                return # data is not a dictionary
            except KeyError:
                continue
            
            if info.function == bpy.props.PointerProperty:
                _cls = info["type"]
                _obj = getattr(obj, key)
                
                BpyProp.deserialize(_obj, data_value, _cls)
            elif info.function == bpy.props.CollectionProperty:
                if not isinstance(data_value, list): continue
                
                _cls = info["type"]
                collection = getattr(obj, key)
                
                while len(collection) != 0:
                    collection.remove(0)
                
                for _data in data_value:
                    _obj = collection.add()
                    BpyProp.deserialize(_obj, _data, _cls)
            else:
                try:
                    setattr(obj, key, data_value)
                except (TypeError, ValueError):
                    pass # wrong data
    
    @staticmethod
    def serialize(obj, cls=None, names=None, use_skip_save=False):
        """Serialize object properties to a JSON data structure"""
        
        if not cls: cls = type(obj)
        
        data = {}
        
        for key, info in BpyProp.iterate(cls, names=names):
            if use_skip_save and ('SKIP_SAVE' in info.get("options", "")): continue
            
            data_value = getattr(obj, key)
            
            if info.function == bpy.props.PointerProperty:
                _cls = info["type"]
                _obj = data_value
                data[key] = BpyProp.serialize(_obj, _cls)
            elif info.function == bpy.props.CollectionProperty:
                _cls = info["type"]
                data[key] = [BpyProp.serialize(_obj, _cls) for _obj in data_value]
            elif info.function in BpyProp.vectors:
                if isinstance(data_value, Matrix): data_value = matrix_flatten(data_value)
                data[key] = list(data_value)
            else:
                data[key] = data_value
        
        return data
    
    # ======================================================================= #
    
    __slots__ = ["function", "keywords"]
    
    def __new__(cls, arg0, arg1=None):
        self = None
        if arg1 is None: # arg0 is potentially a bpy prop
            function, keywords = BpyProp.unwrap(arg0, False)
            if function:
                self = object.__new__(cls)
                self.function, self.keywords = function, keywords
        elif isinstance(arg1, str): # arg0 is object/type, arg1 is property name
            function, keywords = BpyProp.find(arg0, arg1, unwrap=True)
            if function:
                self = object.__new__(cls)
                self.function, self.keywords = function, keywords
        elif BpyProp.is_wrappable(arg0, arg1):
            self = object.__new__(cls)
            self.function, self.keywords = arg0, arg1
        return self
    
    def __call__(self, copy=False):
        """Create an equivalent bpy.props.* structure"""
        return BpyProp.wrap(self.function, self.keywords.copy() if copy else self.keywords)
    
    # Extra attributes are useful to store custom callbacks and various metadata
    # (accessible via type(obj)'s attributes / __annotations__).
    # Right now, Blender apparently doesn't check the exact type of options/tags
    # (though it seems that tags aren't available for abritrary classes).
    
    class __set_extra(set):
        pass
    
    __dummy = object()
    
    @property
    def extra(self):
        options = self.keywords.get("options", None)
        return getattr(options, "extra", None)
    @extra.setter
    def extra(self, value):
        options = self.keywords.get("options", None)
        extra = getattr(options, "extra", None)
        if extra is value: return
        options_extra = options
        if not hasattr(options_extra, "extra"):
            options_extra = self.__set_extra()
            if options is None: options = {'ANIMATABLE'}
            if options: options_extra.update(options)
            self.keywords["options"] = options_extra
        options_extra.extra = value
    
    def get(self, name, default=__dummy):
        result = self.keywords.get(name, default)
        if result is not BpyProp.__dummy: return result
        return BpyProp.known[self.function].get(name)
    
    def __getitem__(self, name):
        return self.keywords[name]
    
    def __setitem__(self, name, value):
        if name == "options":
            extra = self.extra
            if extra is not None:
                value = self.__set_extra(value)
                value.extra = extra
        self.keywords[name] = value
    
    def __contains__(self, name):
        return name in self.keywords
    
    def __len__(self):
        return len(self.keywords)
    
    def __iter__(self):
        return self.keywords.keys()
    
    def items(self):
        return self.keywords.items()
    
    def keys(self):
        return self.keywords.keys()
    
    def values(self):
        return self.keywords.values()
    
    def update(self, *args, **kwargs):
        for arg in args:
            for k, v in (arg.items() if hasattr(arg, "items") else arg):
                self[k] = v
        
        for k, v in kwargs.items():
            self[k] = v
    
    def add_missing(self, d):
        for arg in args:
            for k, v in (arg.items() if hasattr(arg, "items") else arg):
                if k not in self.keywords: self[k] = v
        
        for k, v in kwargs.items():
            if k not in self.keywords: self[k] = v

BpyProp.wrapper_types_extended = (BpyProp.wrapper_type, BpyProp)

class BpyEnum:
    Item = namedtuple("Item", ["identifier", "name", "description", "icon", "value"])
    
    @classmethod
    def normalize_item(cls, item, value=0, wrap=False):
        if item is None:
            return None # separator
        elif isinstance(item, tuple) and (len(item) == 5):
            if wrap: return cls.Item(*item)
            return item
        elif isinstance(item, bpy.types.EnumPropertyItem):
            if wrap: return cls.Item(item.identifier, item.name, item.description, item.icon, item.value)
            return (item.identifier, item.name, item.description, item.icon, item.value)
        elif isinstance(item, str):
            if wrap: return cls.Item(item, item, "", 'NONE', value)
            return (item, item, "", 'NONE', value)
        elif isinstance(item, dict):
            identifier = item.get("identifier", "")
            name = item.get("name", "")
            description = item.get("description", "")
            icon = item.get("icon", 'NONE')
            value = item.get("value", value)
            if wrap: return cls.Item(identifier or name, name or identifier, description, icon, value)
            return (identifier or name, name or identifier, description, icon, value)
        else: # iterable
            item = iter(item)
            identifier = next(item, "")
            name = next(item, identifier)
            description = next(item, "")
            arg4 = next(item, None)
            arg5 = next(item, None)
            if arg5 is None:
                if isinstance(arg4, int):
                    value = arg4
                elif isinstance(arg4, str):
                    icon = arg4
                else:
                    icon = 'NONE'
            else:
                icon, value = arg4, arg5
            if wrap: return cls.Item(identifier, name, description, icon, value)
            return (identifier, name, description, icon, value)
    
    @classmethod
    def normalize_items(cls, items, is_flag, separators=True, container=list, wrap=False):
        return container(cls.normalize_item(item, value, wrap) for value, item in cls.enumerate(items, is_flag, separators))
    
    @classmethod
    def enumerate(cls, items, is_flag, separators=True):
        value = 1
        for item in items:
            if item is None:
                if separators: yield None, None
            else:
                yield value, item
                value = ((value << 1) if is_flag else (value + 1))
    
    @classmethod
    def normalize_value(cls, items, value, is_flag):
        # value is expected to be a valid entry
        if not isinstance(value, int):
            if is_flag:
                value = (({value} if isinstance(value, str) else set(value)) if value else set())
            else:
                if (not value) and items: value = items[0][0]
        return value
    
    @classmethod
    def fix_value(cls, items, value, is_flag=None):
        # Some built-in properties may have invalid defaults.
        # E.g. ColorManagedInputColorspaceSettings.name has
        # 'NONE' as default, but it's not a recognized value.
        if isinstance(value, int):
            item_values = {item[-1] for item in items if item}
            if is_flag:
                mask = 0
                for item_value in item_values:
                    mask |= item_value
                return value & mask
            else:
                for item_value in item_values:
                    if item_value == value: return value
                return (items[0][-1] if items else None)
        else:
            if is_flag is None: is_flag = not isinstance(value, str)
            item_keys = {item[0] for item in items if item}
            if is_flag:
                return item_keys.intersection(value)
            else:
                if value in item_keys: return value
                # Some built-in properties may not even have items (e.g.
                # "sort_method" in the built-in importers/exporters) :-(
                # For no-items enums, default=None seems to not cause errors
                return (items[0][0] if items else None)
    
    @classmethod
    def to_int(cls, items, value, is_flag=None):
        # Expects normalized items
        if is_flag is None: is_flag = not isinstance(value, str)
        if is_flag:
            res = 0
            for item in items:
                if item[0] in value: res |= item[-1]
            return res
        else:
            for item in items:
                if item[0] == value: return item[-1]
            return 0
    
    @classmethod
    def from_int(cls, items, value, is_flag):
        # Expects normalized items
        if is_flag:
            return {item[0] for item in items if item[-1] & value}
        else:
            for item in items:
                if item[-1] == value: return item[0]
            return None
    
    @staticmethod
    def memorizer(func):
        if hasattr(func, "is_enum_memorizer"): return func # already wrapped
        
        # Blender requires EXACTLY THE SAME string objects,
        # otherwise there will be glitches (and possibly crashes?)
        strings = {}
        def checker(item):
            for v in item:
                if isinstance(v, str):
                    s = strings.get(v)
                    if s is None:
                        s = str(v) # copy!
                        strings[s] = s
                    v = s
                yield v
        
        items = []
        def wrapper(self, context):
            items.clear()
            items.extend(tuple(checker(item)) for item in func(self, context))
            return items
        
        wrapper.is_enum_memorizer = True
        wrapper.items = items
        
        return wrapper

# ===== PRIMITIVE ITEMS & PROP ===== #

PrimitiveItem = [
    ("Bool", "Bool", dict()),
    ("Int", "Int", dict()),
    ("Float", "Float", dict()),
    ("String", "String", dict()),
    ("Color", "FloatVector", dict(subtype='COLOR', size=3, min=0.0, max=1.0)),
    ("Euler", "FloatVector", dict(subtype='EULER', size=3)),
    ("Quaternion", "FloatVector", dict(subtype='QUATERNION', size=4)),
    ("Matrix3", "FloatVector", dict(subtype='MATRIX', size=9)),
    ("Matrix4", "FloatVector", dict(subtype='MATRIX', size=16)),
    ("Vector2", "FloatVector", dict(subtype='XYZ', size=2)),
    ("Vector3", "FloatVector", dict(subtype='XYZ', size=3)),
    ("Vector4", "FloatVector", dict(subtype='XYZ', size=4)),
]
PrimitiveItem = type("PrimitiveItem", (object,), {
    n:type(n, (bpy.types.PropertyGroup,), {"__annotations__":{"value":getattr(bpy.props, p+"Property")(**d)}})
    for n, p, d in PrimitiveItem
})
# Seems like class registration MUST be done in global namespace
for item_name in dir(PrimitiveItem):
    if not item_name.startswith("_"):
        bpy.utils.register_class(getattr(PrimitiveItem, item_name))
del item_name

class prop:
    """
    A syntactic sugar for more concise bpy properties declaration.
    By abusing operator overloading, we can declare them
    in a way more similar to conventional attributes. Examples:
    
    pointer_property = SomePropertyGroup | prop()
    
    enum_property = 'A' | prop(items=['A', 'B', 'C'])
    enum_property = {'A', 'B'} | prop(items=['A', 'B', 'C'])
    enum_property = {} | prop(items=['A', 'B', 'C'])
    
    bool_property = True | prop()
    int_property = 5 | prop()
    float_property = 3.14 | prop()
    string_property = "Suzanne" | prop()
    
    bool_vector_property = (True, False) | prop()
    int_vector_property = (5, 7, -3, 0) | prop()
    float_vector_property = (3.14, -0.8, 2.7e-3) | prop()
    float_vector_property = <mathutils instance (Vector, Matrix, etc.)> | prop()
    
    collection_property = [SomePropertyGroup] | prop()
    collection_property = [<primitive value or mathutils instance>] | prop() # will use a predefined property group
    collection_property = [<primitive or mathutils type>] | prop() # will use a predefined property group
    """
    
    def __init__(self, name=None, description=None, **kwargs):
        if name is not None: kwargs["name"] = compress_whitespace(name)
        if description is not None: kwargs["description"] = compress_whitespace(description)
        self.kwargs = kwargs
        self.options = set(self.kwargs.get("options", ('ANIMATABLE',)))
        self.kwargs["options"] = self.options
    
    def __pos__(self): # +prop()
        self.options.add('LIBRARY_EDITABLE')
        return self
    
    def __neg__(self): # -prop()
        self.options.add('HIDDEN')
        self.options.add('SKIP_SAVE')
        self.options.discard('ANIMATABLE')
        return self
    
    def __invert__(self): # ~prop()
        self.options.add('PROPORTIONAL')
        return self
    
    def make(self, value):
        self.options.update(self.kwargs.get("options", ()))
        return self.parse_arguments(value, self.kwargs)
    
    # "|" is the overloadable binary operator
    # with the lowest precedence that can
    # still return non-boolean values
    __ror__ = make
    __rlshift__ = make
    __rrshift__ = make
    
    # ========= ARGUMENT PARSING ========= #
    types_primitive = {
        bool:bpy.props.BoolProperty,
        int:bpy.props.IntProperty,
        float:bpy.props.FloatProperty,
        str:bpy.props.StringProperty,
    }
    
    types_primitive_vector = {
        bool:bpy.props.BoolVectorProperty,
        int:bpy.props.IntVectorProperty,
        float:bpy.props.FloatVectorProperty,
    }
    
    types_float_vector_subtype = {
        Vector:'XYZ',
        Matrix:'MATRIX',
        Euler:'EULER',
        Quaternion:'QUATERNION',
        Color:'COLOR',
    }
    
    types_item_instance = {
        bool:PrimitiveItem.Bool,
        int:PrimitiveItem.Int,
        float:PrimitiveItem.Float,
        str:PrimitiveItem.String,
        Color:PrimitiveItem.Color,
        Euler:PrimitiveItem.Euler,
        Quaternion:PrimitiveItem.Quaternion,
        Matrix:(None, None, PrimitiveItem.Matrix3, PrimitiveItem.Matrix4),
        Vector:(None, PrimitiveItem.Vector2, PrimitiveItem.Vector3, PrimitiveItem.Vector4),
    }
    
    types_item_class = {
        bool:PrimitiveItem.Bool,
        int:PrimitiveItem.Int,
        float:PrimitiveItem.Float,
        str:PrimitiveItem.String,
        Color:PrimitiveItem.Color,
        Euler:PrimitiveItem.Euler,
        Quaternion:PrimitiveItem.Quaternion,
        Matrix:PrimitiveItem.Matrix4,
        Matrix.to_3x3:PrimitiveItem.Matrix3,
        Matrix.to_4x4:PrimitiveItem.Matrix4,
        Vector:PrimitiveItem.Vector3,
        Vector.to_2d:PrimitiveItem.Vector2,
        Vector.to_3d:PrimitiveItem.Vector3,
        Vector.to_4d:PrimitiveItem.Vector4,
    }
    
    def parse_arguments(self, value, kwargs):
        if value is None: value = kwargs.get("default")
        
        value_target = "default"
        vtype = type(value)
        
        err_msg = f"Unexpected property value {repr(value)}; impossible to infer property type"
        
        ptr_types = (bpy.types.PropertyGroup, bpy.types.ID)
        
        if BpyProp.validate(value):
            value_target = None
        elif issubclass_safe(value, ptr_types): # a = SomePG | prop()
            bpy_type = bpy.props.PointerProperty
            value_target = "type"
        elif ("items" in kwargs) and isinstance(value, str): # a = 'A' | prop(items=['A', 'B', 'C'])
            bpy_type = bpy.props.EnumProperty
            value = self.complete_enum_items(kwargs, value)
        elif vtype in self.types_primitive: # a = 1024.0 | prop()
            bpy_type = self.types_primitive[vtype]
        elif vtype in self.types_float_vector_subtype: # a = Vector() | prop()
            bpy_type = bpy.props.FloatVectorProperty
            if vtype is Matrix: value = matrix_flatten(value)
            if "subtype" not in kwargs: kwargs["subtype"] = self.types_float_vector_subtype[vtype]
            if kwargs["subtype"] == 'COLOR': # need to set min-max to 0..1, otherwise glitches
                kwargs.setdefault("min", 0.0)
                kwargs.setdefault("max", 1.0)
                if "alpha" in kwargs: value = (value[0], value[1], value[2], kwargs.pop("alpha"))
            value = tuple(value)
            kwargs["size"] = np.shape(value)
        elif isinstance(value, tuple) and value: # a = (False, False, False) | prop()
            itype = type(value[0])
            if itype in self.types_primitive_vector:
                bpy_type = self.types_primitive_vector[itype]
                if kwargs.get("subtype") == 'COLOR': # need to set min-max to 0..1, otherwise glitches
                    kwargs.setdefault("min", 0.0)
                    kwargs.setdefault("max", 1.0)
                kwargs["size"] = np.shape(value)
            else:
                raise TypeError(err_msg)
        elif isinstance(value, list) and value: # a = [...] | prop()
            bpy_type = bpy.props.CollectionProperty
            value_target = "type"
            item = value[0]
            itype = type(item)
            if isinstance(item, dict): # a = [dict(prop1=..., prop2=..., ...)] | prop()
                value = type(kwargs.get("name", "<Auto PropertyGroup>"),
                    (bpy.types.PropertyGroup,), {"__annotations__":item})
                value.__name__ += ":AUTOREGISTER" # for AddonManager
            elif itype in self.types_item_instance: # a = [Matrix()] | prop()
                value = self.types_item_instance[itype]
                if not isinstance(value, type): value = value[len(item)]
            elif issubclass_safe(item, ptr_types): # a = [SomePG] | prop()
                value = item
            elif item in self.types_item_class: # a = [Matrix] | prop()
                value = self.types_item_class[item]
            else:
                raise TypeError(err_msg)
        elif isinstance(value, set): # a = {'A', 'B'} | prop(items=['A', 'B', 'C'])
            bpy_type = bpy.props.EnumProperty
            value = self.complete_enum_items(kwargs, value, True)
        elif isinstance(value, dict): # a = {...} | prop()
            if "items" not in kwargs: # a = dict(prop1=..., prop2=..., ...) | prop()
                bpy_type = bpy.props.PointerProperty
                value_target = "type"
                value = type(kwargs.get("name", "<Auto PropertyGroup>"),
                    (bpy.types.PropertyGroup,), {"__annotations__":value})
                value.__name__ += ":AUTOREGISTER" # for AddonManager
            elif not value: # a = {} | prop(items=['A', 'B', 'C'])
                bpy_type = bpy.props.EnumProperty
                value = self.complete_enum_items(kwargs, value, True)
            else:
                raise TypeError(err_msg)
        else:
            raise TypeError(err_msg)
        
        if value_target is not None:
            if value is None:
                kwargs.pop(value_target, None)
            else:
                kwargs[value_target] = value
            prop_info = BpyProp(bpy_type, {})
        else:
            prop_info = BpyProp(value)
        
        if "extra" in kwargs:
            prop_info.extra = kwargs["extra"]
            del kwargs["extra"]
        
        prop_info.update(kwargs)
        
        if prop_info.function == bpy.props.EnumProperty:
            items = prop_info.get("items")
            if callable(items): prop_info["items"] = BpyEnum.memorizer(items)
        
        return prop_info()
    
    @classmethod
    def complete_enum_items(cls, kwargs, value, enum_flag=False):
        options = kwargs.get("options")
        if options and ('ENUM_FLAG' in options):
            enum_flag = True
        elif enum_flag:
            kwargs.setdefault("options", set()).add('ENUM_FLAG')
        
        items = kwargs.get("items", ())
        
        if hasattr(items, "__iter__"): # sequence -> ensure full form
            # Note: empty items is a valid situation (e.g. when they are populated later)
            
            normalized_items = []
            for item_value, item in BpyEnum.enumerate(items, enum_flag):
                identifier, name, description, icon, item_value = BpyEnum.normalize_item(item, item_value)
                name = compress_whitespace(name)
                description = compress_whitespace(description)
                normalized_items.append((identifier, name, description, icon, item_value))
            
            kwargs["items"] = normalized_items
            
            value = BpyEnum.normalize_value(normalized_items, value, enum_flag)
        else: # function/callback -> default value should be None (?)
            value = None
        
        return value

# ===== FILE FILTERS ===== #

# Correspondence between use_filter_* in Python and FILE_TYPE_* in C: rna_def_fileselect_params()
# https://developer.blender.org/diffusion/B/browse/master/source/blender/makesrna/intern/rna_space.c

# Blender-file extensions: BLO_has_bfile_extension()
# https://developer.blender.org/diffusion/B/browse/master/source/blender/blenloader/intern/readfile.c

# Mapping between file extensions and FILE_TYPE_*: ED_path_extension_type()
# Checking if file name is a .blend backup: file_is_blend_backup()
# Mapping between FILE_TYPE_* and icons: filelist_geticon_ex()
# https://developer.blender.org/diffusion/B/browse/master/source/blender/editors/space_file/filelist.c

# Lists of file extensions for some file categories:
# https://developer.blender.org/diffusion/B/browse/master/source/blender/imbuf/intern/util.c

bpy_file_filters = { # in the order defined in ED_path_extension_type()
    "blender": [ # use_filter_blender, FILE_TYPE_BLENDER
        ".blend", ".ble", ".blend.gz", # BLO_has_bfile_extension()
    ],
    "backup": [ # use_filter_backup, FILE_TYPE_BLENDER_BACKUP
        ".blend?", ".blend??", # file_is_blend_backup()
    ],
    "app_bundle": [ # (no browser filter), FILE_TYPE_APPLICATIONBUNDLE
        ".app",
    ],
    "script": [ # use_filter_script, FILE_TYPE_PYSCRIPT
        ".py",
    ],
    "text": [ # use_filter_text, FILE_TYPE_TEXT
        ".txt", ".glsl", ".osl", ".data", ".pov", ".ini", ".mcr", ".inc",
    ],
    "font": [ # use_filter_font, FILE_TYPE_FTFONT
        ".ttf", ".ttc", ".pfb", ".otf", ".otc",
    ],
    "btx": [ # (no browser filter), FILE_TYPE_BTX
        ".btx",
    ],
    "collada": [ # (no browser filter), FILE_TYPE_COLLADA
        ".dae",
    ],
    "alembic": [ # (no browser filter), FILE_TYPE_ALEMBIC
        ".abc",
    ],
    "image": [ # use_filter_image, FILE_TYPE_IMAGE
	    ".png", ".tga",  ".bmp", ".jpg", ".jpeg", ".sgi", ".rgb", ".rgba",
	    ".tif", ".tiff", ".tx", # WITH_TIFF
	    ".jp2", ".j2c", # WITH_OPENJPEG
	    ".hdr", # WITH_HDR
	    ".dds", # WITH_DDS
	    ".dpx", ".cin", # WITH_CINEON
	    ".exr", # WITH_OPENEXR
	    ".psd", ".pdd",  ".psb", # WITH_OPENIMAGEIO
    ],
    "movie": [ # use_filter_movie, FILE_TYPE_MOVIE
	    ".avi", ".flc", ".mov", ".movie", ".mp4", ".m4v", ".m2v", ".m2t", ".m2ts", ".mts",
	    ".ts", ".mv", ".avs", ".wmv", ".ogv", ".ogg", ".r3d", ".dv", ".mpeg", ".mpg",
	    ".mpg2", ".vob", ".mkv", ".flv", ".divx", ".xvid", ".mxf", ".webm",
    ],
    "sound": [ # use_filter_sound, FILE_TYPE_SOUND
	    ".wav", ".ogg", ".oga", ".mp3", ".mp2", ".ac3", ".aac",
        ".flac", ".wma", ".eac3", ".aif", ".aiff", ".m4a", ".mka",
    ],
    "folder": [ # use_filter_folder, FILE_TYPE_FOLDER
    ],
    "blendid": [ # use_filter_blendid, FILE_TYPE_BLENDERLIB
    ],
}

bpy_file_icons = {
    "blender": 'FILE_BLEND',
    "backup": 'FILE_BACKUP',
    "app_bundle": 'UGLYPACKAGE',
    "script": 'FILE_SCRIPT',
    "text": 'FILE_TEXT',
    "font": 'FILE_FONT',
    "image": 'FILE_IMAGE',
    "movie": 'FILE_MOVIE',
    "sound": 'FILE_SOUND',
    "folder": 'FILE_FOLDER',
} # all others: FILE_BLANK / FILE

# ===== ENUMS ===== #

class ObjectModeInfo:
    context_modes = {item.identifier for item in bpy.types.Context.bl_rna.properties["mode"].enum_items}
    object_modes = {item.identifier for item in bpy.types.Object.bl_rna.properties["mode"].enum_items}
    
    def __bool__(self):
        return (self.context in self.context_modes) and (self.object in self.object_modes)
    
    def __init__(self, context, object, type, subtype, multi_object):
        self.context = context
        self.object = object
        self.type = type
        self.subtype = subtype
        self.multi_object = multi_object
        
        name = (object if len(object) >= len(context) else context)
        name = name.replace("GPENCIL", "GREASE_PENCIL")
        self.name = " ".join([word.capitalize() for word in name.split("_")])

class ObjectTypeInfo:
    object_types = {item.identifier for item in bpy.types.Object.bl_rna.properties["type"].enum_items}
    
    def __bool__(self):
        return self.name in self.object_types
    
    def __init__(self, name, data_name, features, modes, conversions=None):
        self.name = name
        self.data_name = data_name
        self.data_type = (getattr(bpy.types, self.data_name, None) if self.data_name else None)
        self.features = features
        self.modes = self.parse_modes(modes)
        self.context_modes = {mode.context: mode for mode in self.modes}
        self.object_modes = {mode.object: mode for mode in self.modes}
        self.conversions = self.parse_conversions(conversions)
    
    @classmethod
    def parse_conversions(cls, conversions):
        if not conversions: return {}
        
        version = (3, 6, 0)
        
        if isinstance(conversions, dict):
            return {key: (version if isinstance(value, bool) else value)
                    for key, value in conversions.items()
                    if key in cls.object_types}
        
        return {key: version for key in conversions if key in cls.object_types}
    
    @classmethod
    def parse_modes(cls, modes):
        result = []
        
        for mode_info in modes:
            if isinstance(mode_info, dict):
                mode_info = ObjectModeInfo(**mode_info)
            elif not isinstance(mode_info, ObjectModeInfo):
                mode_info = ObjectModeInfo(*mode_info)
            
            if mode_info: result.append(mode_info)
        
        return result
    
    def is_convertible(self, obj_type):
        blender_version = self.conversions.get(obj_type)
        return (False if blender_version is None else bpy.app.version >= blender_version)

class BlEnums:
    extensible_classes = (bpy.types.PropertyGroup, bpy.types.ID, bpy.types.Bone, bpy.types.PoseBone) # see bpy_struct documentation
    
    common_attrs = {'bl_idname', 'bl_label', 'bl_description', 'bl_options',
        'bl_context', 'bl_region_type', 'bl_space_type', 'bl_category',
        'bl_use_postprocess', 'bl_use_preview', 'bl_use_shading_nodes',
        'is_animation', 'is_preview',
        'bl_width_default', 'bl_width_max', 'bl_width_min',
        'bl_height_default', 'bl_height_max', 'bl_height_min'}
    
    options = {type_name: {item.identifier for item in getattr(bpy.types, type_name).bl_rna.properties["bl_options"].enum_items}
        for type_name in ("GizmoGroup", "KeyingSetInfo", "Macro", "Operator", "Panel")
        if "bl_options" in getattr(bpy.types, type_name).bl_rna.properties}
    
    space_types = {item.identifier for item in bpy.types.Space.bl_rna.properties["type"].enum_items}
    region_types = {item.identifier for item in bpy.types.Region.bl_rna.properties["type"].enum_items}
    
    context_modes = ObjectModeInfo.context_modes
    object_modes = ObjectModeInfo.object_modes
    object_types = ObjectTypeInfo.object_types
    
    @classmethod
    def mode_from_object(cls, obj):
        if not obj: return 'OBJECT'
        return cls.object_infos[obj.type].object_modes[obj.mode].context
    
    @classmethod
    def mode_to_object(cls, context_mode):
        mode = cls.mode_infos.get(context_mode)
        return (mode.object if mode else None)
    
    @classmethod
    def normalize_mode(cls, mode, obj=None):
        if mode in cls.context_modes: return mode
        if not obj: return None
        mode = cls.object_infos[obj.type].object_modes.get(obj.mode)
        return (mode.context if mode else None)
    
    @classmethod
    def is_mode_valid(cls, mode, obj):
        if not obj: return mode == 'OBJECT'
        object_info = cls.object_infos[obj.type]
        return (mode in object_info.object_modes) or (mode in object_info.context_modes)
    
    @classmethod
    def get_mode_name(cls, mode):
        mode = mode.replace("GPENCIL", "GREASE_PENCIL")
        return " ".join([word.capitalize() for word in mode.split("_")])
    
    @classmethod
    def is_convertible(cls, src_type, dst_type):
        src_info = cls.object_infos.get(src_type)
        dst_info = cls.object_infos.get(dst_type)
        return (src_info.is_convertible(dst_info.name) if src_info and dst_info else False)
    
    # Panel.bl_context is not an enum property, so we can't get all possible values through introspection
    panel_contexts = {
        'VIEW_3D':{
            "mesh_edit":'EDIT_MESH',
            "curve_edit":'EDIT_CURVE',
            "surface_edit":'EDIT_SURFACE',
            "text_edit":'EDIT_TEXT',
            "armature_edit":'EDIT_ARMATURE',
            "mball_edit":'EDIT_METABALL',
            "lattice_edit":'EDIT_LATTICE',
            "posemode":'POSE',
            "sculpt_mode":'SCULPT',
            "weightpaint":'PAINT_WEIGHT',
            "vertexpaint":'PAINT_VERTEX',
            "texturepaint":'PAINT_TEXTURE',
            "particlemode":'PARTICLE',
            "objectmode":'OBJECT',
        },
        'PROPERTIES':{
            "object", "data", "material", "physics", "constraint",
            "particle", "render", "scene", "texture", "world"
        },
    }

# Due to python's scope idiosyncrasies, these attributes are easier to define outside of the class
BlEnums.object_infos = [
    ObjectTypeInfo('MESH', "Mesh", {'RENDERABLE':True, 'MODIFIERS':True}, [
        ('OBJECT', 'OBJECT', 'OBJECT', None, True),
        ('EDIT_MESH', 'EDIT', 'EDIT', 'MESH', True),
        ('SCULPT', 'SCULPT', 'SCULPT', None, False),
        ('PAINT_WEIGHT', 'WEIGHT_PAINT', 'PAINT', 'WEIGHT', False),
        ('PAINT_VERTEX', 'VERTEX_PAINT', 'PAINT', 'VERTEX', False),
        ('PAINT_TEXTURE', 'TEXTURE_PAINT', 'PAINT', 'TEXTURE', False),
        ('PARTICLE', 'PARTICLE_EDIT', 'EDIT', 'PARTICLE', False),
    ], {'MESH':True, 'GPENCIL':True, 'GREASEPENCIL':(4,3,0), 'POINTCLOUD':True}),
    ObjectTypeInfo('CURVE', "Curve", {'RENDERABLE':True, 'MODIFIERS':True}, [
        ('OBJECT', 'OBJECT', 'OBJECT', None, True),
        ('EDIT_CURVE', 'EDIT', 'EDIT', 'CURVE', True),
    ], {'MESH':True, 'GPENCIL':True, 'GREASEPENCIL':(4,3,0), 'CURVES':True}),
    ObjectTypeInfo('SURFACE', "SurfaceCurve", {'RENDERABLE':True, 'MODIFIERS':True}, [
        ('OBJECT', 'OBJECT', 'OBJECT', None, True),
        ('EDIT_SURFACE', 'EDIT', 'EDIT', 'SURFACE', True),
    ], {'MESH':True}),
    ObjectTypeInfo('CURVES', "Curves", {'RENDERABLE':True, 'MODIFIERS':True}, [
        ('OBJECT', 'OBJECT', 'OBJECT', None, True),
        ('EDIT_CURVES', 'EDIT', 'EDIT', 'CURVES', True),
        ('SCULPT_CURVES', 'SCULPT', 'SCULPT', None, False),
    ], {'MESH':True}),
    ObjectTypeInfo('META', "MetaBall", {'RENDERABLE':True}, [
        ('OBJECT', 'OBJECT', 'OBJECT', None, True),
        ('EDIT_METABALL', 'EDIT', 'EDIT', 'META', True),
    ], {'MESH':True}),
    ObjectTypeInfo('FONT', "TextCurve", {'RENDERABLE':True, 'MODIFIERS':True}, [
        ('OBJECT', 'OBJECT', 'OBJECT', None, True),
        ('EDIT_TEXT', 'EDIT', 'EDIT', 'FONT', True),
    ], {'MESH':True, 'CURVE':True}),
    ObjectTypeInfo('POINTCLOUD', "PointCloud", {'RENDERABLE':True, 'MODIFIERS':True}, [
        ('OBJECT', 'OBJECT', 'OBJECT', None, True),
        ('EDIT_POINT_CLOUD', 'EDIT', 'EDIT', 'POINTCLOUD', True),
    ], {'MESH':True}),
    ObjectTypeInfo('VOLUME', "Volume", {'RENDERABLE':True, 'MODIFIERS':True}, [
        ('OBJECT', 'OBJECT', 'OBJECT', None, True),
    ]),
    ObjectTypeInfo('GPENCIL', "GreasePencil", {'RENDERABLE':True, 'MODIFIERS':True}, [
        ('OBJECT', 'OBJECT', 'OBJECT', None, True),
        ('PAINT_GPENCIL', 'PAINT_GPENCIL', 'DRAW', None, False),
        ('EDIT_GPENCIL', 'EDIT_GPENCIL', 'EDIT', 'GPENCIL', False),
        ('SCULPT_GPENCIL', 'SCULPT_GPENCIL', 'SCULPT', None, False),
        ('WEIGHT_GPENCIL', 'WEIGHT_GPENCIL', 'PAINT', 'WEIGHT', False),
        ('VERTEX_GPENCIL', 'VERTEX_GPENCIL', 'PAINT', 'VERTEX', False),
    ]),
    ObjectTypeInfo('GREASEPENCIL', "GreasePencilv3", {'RENDERABLE':True, 'MODIFIERS':True}, [
        ('OBJECT', 'OBJECT', 'OBJECT', None, True),
        ('EDIT_GREASE_PENCIL', 'EDIT', 'EDIT', 'GREASEPENCIL', False),
        ('SCULPT_GREASE_PENCIL', 'SCULPT_GREASE_PENCIL', 'SCULPT', 'GREASEPENCIL', False),
        ('PAINT_GREASE_PENCIL', 'PAINT_GREASE_PENCIL', 'DRAW', None, False),
        ('WEIGHT_GREASE_PENCIL', 'WEIGHT_GREASE_PENCIL', 'PAINT', 'WEIGHT', False),
        ('VERTEX_GREASE_PENCIL', 'VERTEX_GREASE_PENCIL', 'PAINT', 'VERTEX', False),
    ]),
    ObjectTypeInfo('ARMATURE', "Armature", {}, [
        ('OBJECT', 'OBJECT', 'OBJECT', None, True),
        ('EDIT_ARMATURE', 'EDIT', 'EDIT', 'ARMATURE', True),
        ('POSE', 'POSE', 'POSE', None, True),
    ]),
    ObjectTypeInfo('LATTICE', "Lattice", {'MODIFIERS':True}, [
        ('OBJECT', 'OBJECT', 'OBJECT', None, True),
        ('EDIT_LATTICE', 'EDIT', 'EDIT', 'LATTICE', True),
    ]),
    ObjectTypeInfo('EMPTY', None, {'RENDERABLE':True}, [
        ('OBJECT', 'OBJECT', 'OBJECT', None, True),
    ]),
    ObjectTypeInfo('LIGHT', "Light", {'RENDERABLE':True}, [
        ('OBJECT', 'OBJECT', 'OBJECT', None, True),
    ]),
    ObjectTypeInfo('LIGHT_PROBE', "LightProbe", {'RENDERABLE':True}, [
        ('OBJECT', 'OBJECT', 'OBJECT', None, True),
    ]),
    ObjectTypeInfo('CAMERA', "Camera", {'RENDERABLE':True}, [
        ('OBJECT', 'OBJECT', 'OBJECT', None, True),
    ]),
    ObjectTypeInfo('SPEAKER', "Speaker", {'RENDERABLE':True}, [
        ('OBJECT', 'OBJECT', 'OBJECT', None, True),
    ]),
]
BlEnums.object_infos = {info.name: info for info in BlEnums.object_infos if info.name in BlEnums.object_types}
BlEnums.mode_infos = {mode.context: mode for info in BlEnums.object_infos.values() for mode in info.modes}
BlEnums.context_modes_safe = {mode for mode in BlEnums.context_modes if mode in BlEnums.mode_infos}

#============================================================================#

def scan_bpy_types():
    # Exceptions may happen due to bugs in Blender
    for name in dir(bpy.types):
        try:
            bpy_type = getattr(bpy.types, name)
        except Exception as exc:
            print(f"bpy.types.{name}: getattr(bpy.types, {repr(name)}) failed: {exc}")
            continue
        
        try:
            rna_type = getattr(bpy_type, "bl_rna", None)
        except Exception as exc:
            print(f"bpy.types.{name}: getattr({bpy_type}, 'bl_rna', None) failed: {exc}")
            rna_type = None
        
        try:
            is_class = inspect.isclass(bpy_type)
        except Exception as exc:
            print(f"bpy.types.{name}: inspect.isclass({bpy_type}) failed: {exc}")
            is_class = None
        
        yield (name, bpy_type, rna_type, is_class)

# Use this only for built-in classes
bpy_types = {name:bpy_type for name, bpy_type, rna_type, is_class in scan_bpy_types() if is_class}

class IDTypes:
    _maps = {}
    _sorted_keys = {}
    all = []
    all_types = {}
    
    @classmethod
    def register(cls, **kwargs):
        info = IDTypes()
        for key, value in kwargs.items():
            setattr(info, key, value)
        cls.all.append(info)
    
    @classmethod
    def post_register(cls, validate=False):
        cls.all_types = {subtype.bl_rna.identifier:info for info in IDTypes.all for subtype in info.types}
        
        if not validate: return
        
        registered_types = {t for key in cls.map("types").keys() for t in key}
        all_id_types = {v for v in bpy_types.values() if issubclass(v, bpy.types.ID)}
        print("ID types not present in IDTypes:", all_id_types - registered_types)
    
    @classmethod
    def map(cls, attr_name):
        attr_map = cls._maps.get(attr_name)
        if attr_map is not None: return attr_map
        attr_map = {}
        for info in cls.all:
            value = getattr(info, attr_name)
            if (value is None) or (value in attr_map) or (info.type is None): continue
            attr_map[value] = info
        cls._maps[attr_name] = attr_map
        return attr_map
    
    @classmethod
    def keys(cls, attr_name):
        keys = cls._sorted_keys.get(attr_name)
        if keys is not None: return keys
        keys = sorted(cls.map(attr_name).keys())
        cls._sorted_keys[attr_name] = keys
        return keys
    
    def __getattr__(self, key):
        return None
    
    @property
    def type(self):
        # No point in caching, this is very simple already
        return bpy_types.get(self.type_name)
    
    @property
    def types(self):
        if "_types" in self.__dict__: return self._types
        base_type = self.type
        if not base_type:
            self._types = ()
            return self._types
        self._types = tuple(target_type for target_type in bpy_types.values() if issubclass(target_type, base_type))
        return self._types
    
    @property
    def data(self):
        # Note: during addon initialization/registration, bpy.data
        # returns bpy_restrict_state._RestrictData object instead
        # of bpy.types.BlendData
        if "_data" in self.__dict__: return self._data
        bpy_data = bpy.data
        if not isinstance(bpy_data, bpy_types["BlendData"]): return None
        data_name = self.data_name # can be None
        self._data = (getattr(bpy_data, data_name, None) if data_name else None)
        return self._data

# https://docs.blender.org/api/current/bpy.types.BlendData.html
# https://docs.blender.org/api/current/bpy_types_enum_items/id_type_items.html
# https://docs.blender.org/api/current/bpy_types_enum_items/object_type_items.html
IDTypes.register(type_name="Action", data_name="actions", id_type='ACTION',
    label="Action", icon='ACTION')
IDTypes.register(type_name="Armature", data_name="armatures", id_type='ARMATURE',
    label="Armature", icon='ARMATURE_DATA', obj_type='ARMATURE')
IDTypes.register(type_name="Brush", data_name="brushes", id_type='BRUSH',
    label="Brush", icon='BRUSH_DATA')
IDTypes.register(type_name="CacheFile", data_name="cache_files", id_type='CACHEFILE',
    label="Cache File", icon='FILE_CACHE')
IDTypes.register(type_name="Camera", data_name="cameras", id_type='CAMERA',
    label="Camera", icon='CAMERA_DATA', obj_type='CAMERA')
IDTypes.register(type_name="Collection", data_name="collections", id_type='COLLECTION',
    label="Collection", icon='OUTLINER_COLLECTION')
IDTypes.register(type_name="Curve", data_name="curves", id_type='CURVE',
    label="Curve", icon='CURVE_DATA', obj_type='CURVE')
IDTypes.register(type_name="SurfaceCurve", data_name="curves", id_type='CURVE',
    label="Surface Curve", icon='SURFACE_DATA', obj_type='SURFACE')
IDTypes.register(type_name="TextCurve", data_name="curves", id_type='CURVE',
    label="Text Curve", icon='FONT_DATA', obj_type='FONT')
IDTypes.register(type_name="Curves", data_name="hair_curves", id_type='CURVES',
    label="Hair Curves", icon='CURVES_DATA', obj_type='CURVES',)
IDTypes.register(type_name="VectorFont", data_name="fonts", id_type='FONT',
    label="Font", icon='FILE_FONT')
IDTypes.register(type_name="GreasePencil", data_name="grease_pencils", id_type='GREASEPENCIL',
    label="Grease Pencil", icon='GREASEPENCIL', obj_type='GPENCIL')
IDTypes.register(type_name="GreasePencilv3", data_name="grease_pencils_v3", id_type='GREASEPENCIL_V3',
    label="Grease Pencil v3", icon='GREASEPENCIL', obj_type='GREASEPENCIL')
IDTypes.register(type_name="Image", data_name="images", id_type='IMAGE',
    label="Image", icon='IMAGE_DATA')
IDTypes.register(type_name="Key", data_name="shape_keys", id_type='KEY',
    label="Key", icon='SHAPEKEY_DATA')
IDTypes.register(type_name="Lattice", data_name="lattices", id_type='LATTICE',
    label="Lattice", icon='LATTICE_DATA', obj_type='LATTICE')
IDTypes.register(type_name="Library", data_name="libraries", id_type='LIBRARY',
    label="Library", icon='LIBRARY_DATA_DIRECT')
IDTypes.register(type_name="Light", data_name="lights", id_type='LIGHT',
    label="Light", icon='LIGHT', obj_type='LIGHT')
IDTypes.register(type_name="LightProbe", data_name="lightprobes", id_type='LIGHT_PROBE',
    label="Light Probe", icon='OUTLINER_DATA_LIGHTPROBE', obj_type='LIGHT_PROBE')
IDTypes.register(type_name="FreestyleLineStyle", data_name="linestyles", id_type='LINESTYLE',
    label="Line Style", icon='LINE_DATA')
IDTypes.register(type_name="Mask", data_name="masks", id_type='MASK',
    label="Mask", icon='MOD_MASK')
IDTypes.register(type_name="Material", data_name="materials", id_type='MATERIAL',
    label="Material", icon='MATERIAL')
IDTypes.register(type_name="Mesh", data_name="meshes", id_type='MESH',
    label="Mesh", icon='MESH_DATA', obj_type='MESH')
IDTypes.register(type_name="MetaBall", data_name="metaballs", id_type='META',
    label="Metaball", icon='META_DATA', obj_type='META')
IDTypes.register(type_name="MovieClip", data_name="movieclips", id_type='MOVIECLIP',
    label="Movie Clip", icon='FILE_MOVIE')
IDTypes.register(type_name="NodeTree", data_name="node_groups", id_type='NODETREE',
    label="Node Tree", icon='NODETREE')
IDTypes.register(type_name="Object", data_name="objects", id_type='OBJECT',
    label="Object", icon='OBJECT_DATA')
IDTypes.register(type_name="PaintCurve", data_name="paint_curves", id_type='PAINTCURVE',
    label="Paint Curve", icon='CURVE_BEZCURVE')
IDTypes.register(type_name="Palette", data_name="palettes", id_type='PALETTE',
    label="Palette", icon='COLOR')
IDTypes.register(type_name="ParticleSettings", data_name="particles", id_type='PARTICLE',
    label="Particle", icon='PARTICLES')
IDTypes.register(type_name="PointCloud", data_name="pointclouds", id_type='POINTCLOUD',
    label="Point Cloud", icon='POINTCLOUD_DATA', obj_type='POINTCLOUD')
IDTypes.register(type_name="Scene", data_name="scenes", id_type='SCENE',
    label="Scene", icon='SCENE_DATA')
IDTypes.register(type_name="Screen", data_name="screens", id_type='SCREEN',
    label="Screen", icon='SCREEN_BACK', broken={"idprop":True})
IDTypes.register(type_name="Sound", data_name="sounds", id_type='SOUND',
    label="Sound", icon='SOUND')
IDTypes.register(type_name="Speaker", data_name="speakers", id_type='SPEAKER',
    label="Speaker", icon='SPEAKER', obj_type='SPEAKER')
IDTypes.register(type_name="Text", data_name="texts", id_type='TEXT',
    label="Text", icon='TEXT')
IDTypes.register(type_name="Texture", data_name="textures", id_type='TEXTURE',
    label="Texture", icon='TEXTURE')
IDTypes.register(type_name="Volume", data_name="volumes", id_type='VOLUME',
    label="Volume", icon='VOLUME_DATA', obj_type='VOLUME')
IDTypes.register(type_name="WindowManager", data_name="window_managers", id_type='WINDOWMANAGER',
    label="Window Manager", icon='WINDOW')
IDTypes.register(type_name="WorkSpace", data_name="workspaces", id_type='WORKSPACE',
    label="Workspace", icon='WORKSPACE')
IDTypes.register(type_name="World", data_name="worlds", id_type='WORLD',
    label="World", icon='WORLD')
IDTypes.register(label="Empty", icon='EMPTY_DATA', obj_type='EMPTY')
IDTypes.post_register()

class BpyData:
    bpy_type_to_data = {}
    bpy_data_to_type = {}
    
    @classmethod
    def initialize(cls):
        for info in IDTypes.map("data_name").values():
            type_name, data_name = info.type_name, info.data_name
            
            cls.bpy_type_to_data[type_name] = data_name
            cls.bpy_data_to_type[data_name] = type_name
            
            for bpy_type in info.types:
                type_name = bpy_type.bl_rna.identifier
                cls.bpy_type_to_data[type_name] = data_name
    
    @classmethod
    def type_names(cls):
        if not cls.bpy_type_to_data: cls.initialize()
        return cls.bpy_type_to_data.keys()
    
    @classmethod
    def data_names(cls):
        if not cls.bpy_type_to_data: cls.initialize()
        return cls.bpy_data_to_type.keys()
    
    @classmethod
    def get_type_name(cls, data_name):
        if not cls.bpy_type_to_data: cls.initialize()
        return cls.bpy_data_to_type.get(data_name)
    
    @classmethod
    def get_data_name(cls, bpy_type):
        if not cls.bpy_type_to_data: cls.initialize()
        if not isinstance(bpy_type, str): bpy_type = bpy_type.bl_rna.identifier
        return cls.bpy_type_to_data.get(bpy_type)
    
    @classmethod
    def to_data(cls, bpy_type):
        # Since some version of Blender 2.8 beta, bpy.data is empty
        # (not populated) during addon loading (maybe Blender bug?)
        if not cls.bpy_type_to_data: cls.initialize()
        if not isinstance(bpy_type, str): bpy_type = bpy_type.bl_rna.identifier
        return getattr(bpy.data, cls.bpy_type_to_data[bpy_type])

#============================================================================#

def make_idprops_converters():
    def to_bool(value):
        try:
            return bool(value)
        except:
            return None
    
    def to_float(value):
        try:
            return float(value)
        except:
            return None
    
    def safe_round(value):
        if value <= -0x80000000: return -0x80000000
        if value >= 0x7FFFFFFF: return 0x7FFFFFFF
        return (0 if math.isnan(value) else round(value))
    
    def to_int(value):
        if isinstance(value, int): return value
        if isinstance(value, bool): return int(value)
        if isinstance(value, str): value = to_float(value)
        return (None if value is None else safe_round(value))
    
    def to_list(value, ref_value):
        if not isinstance(ref_value, list): ref_value = [ref_value]
        
        ref_type = type(ref_value[0])
        converter = (to_int if ref_type is int else ref_type)
        n = len(value)
        
        if not isinstance(value, list):
            value = converter(value)
            return (ref_value if value is None else [value] * n)
        
        def get(i):
            if i >= n: return ref_value[i]
            if value[i] is None: return ref_value[i]
            v = converter(value[i])
            return (ref_value[i] if v is None else v)
        return [get(i) for i in range(len(ref_value))]
    
    # (source type, target type) : list, bool, int, float, str; dict is N/A
    converters = {
        (list, bool): (lambda value, ref_value: ref_value if value[0] is None else bool(value[0])),
        (list, int): (lambda value, ref_value: ref_value if value[0] is None else to_int(value[0])),
        (list, float): (lambda value, ref_value: ref_value if value[0] is None else float(value[0])),
        (list, str): (lambda value, ref_value: ref_value if value[0] is None else str(value[0])),
        (list, list): to_list,
        (bool, list): to_list,
        (int, list): to_list,
        (float, list): to_list,
        (str, list): to_list,
        (bool, bool): (lambda value, ref_value: value),
        (bool, int): (lambda value, ref_value: int(value)),
        (bool, float): (lambda value, ref_value: float(value)),
        (bool, str): (lambda value, ref_value: str(value)),
        (int, bool): (lambda value, ref_value: bool(value)),
        (int, int): (lambda value, ref_value: value),
        (int, float): (lambda value, ref_value: float(value)),
        (int, str): (lambda value, ref_value: str(value)),
        (float, bool): (lambda value, ref_value: bool(value)),
        (float, int): (lambda value, ref_value: to_int(value)),
        (float, float): (lambda value, ref_value: value),
        (float, str): (lambda value, ref_value: str(value)),
        (str, bool): (lambda value, ref_value: to_bool(value)),
        (str, int): (lambda value, ref_value: to_int(value)),
        (str, float): (lambda value, ref_value: to_float(value)),
        (str, str): (lambda value, ref_value: value),
    }
    
    return converters

class IDProp:
    supports_enum = (bpy.app.version >= (4, 1, 0))
    
    type_map = {bool: "bool", int: "int", float: "float", str: "string"}
    typecode_map = {"b": "bool", "i": "int", "f": "float", "d": "float"}
    
    float_subtypes = [
        ('NONE', "Plain Data", "Data values without special behavior"),
        ('PIXEL', "Pixel", "Pixel"),
        ('PERCENTAGE', "Percentage", "Percentage"),
        ('FACTOR', "Factor", "Factor"),
        ('ANGLE', "Angle", "Angle"),
        ('TIME_ABSOLUTE', "Time", "Time specified in seconds"),
        ('DISTANCE', "Distance", "Distance"),
        ('POWER', "Power", "Power"),
        ('TEMPERATURE', "Temperature", "Temperature"),
    ]
    
    float_array_subtypes = [
        ('NONE', "Plain Data", "Data values without special behavior"),
        ('COLOR', "Linear Color", "Color in the linear space"),
        ('COLOR_GAMMA', "Gamma-Corrected Color", "Color in the gamma corrected space"),
        ('EULER', "Euler Angles", "Euler rotation angles in radians"),
        ('QUATERNION', "Quaternion Rotation", "Quaternion rotation"),
    ]
    
    subtype_names = {
        ("float", False): {item[0] for item in float_subtypes},
        ("float", True): {item[0] for item in float_array_subtypes},
    }
    
    id_types = [IDTypes.map("id_type")[id_type] for id_type in IDTypes.keys("id_type")]
    id_types_broken = {info:bool((info.broken or {}).get("idprop")) for info in IDTypes.all}
    
    converters = make_idprops_converters()
    
    types_with_metadata = (bool, int, float, str, IDPropertyArray, type(None), bpy.types.ID)
    
    def _get(self):
        return self.key in self.container
    exists = property(_get)
    
    def _get(self):
        return self.normalize(self.container[self.key])
    def _set(self, value):
        self.container[self.key] = self.normalize(value)
    value = property(_get, _set)
    
    def _get(self):
        return self.get_type_info(self.container[self.key])
    type_info = property(_get)
    
    def _get(self):
        return self.get_metadata(self.container, self.key)
    def _set(self, value):
        self.set_metadata(self.container, self.key, value)
    metadata = property(_get, _set)
    
    def _get(self):
        return self.get_overridable(self.container, self.key)
    def _set(self, value):
        self.set_overridable(self.container, self.key, value)
    overridable = property(_get, _set)
    
    def _get(self):
        return self.get_data(self.container, self.key)
    def _set(self, value):
        self.set_data(self.container, self.key, value)
    data = property(_get, _set)
    
    def _get(self):
        # Adapted from Blender's rna_prop_ui.py
        rna_prop = self.container.bl_rna.properties.get(self.key)
        return (rna_prop.is_runtime if rna_prop else None)
    is_runtime = property(_get)
    
    def __init__(self, container, key):
        self.container = container
        self.key = key
    
    @classmethod
    def get_type_info(cls, value, string_arrays=False):
        # Blender actually can store the following kinds of properties:
        # * With metadata support:
        #   * Basic types (bool, int, float, str)
        #   * IDPropertyArray (equivalent of bool[], int[], float[])
        #   * bpy.types.ID descendants / None (though only at top-level or in dicts, but not in lists)
        # * Without metadata support:
        #   * Python lists of objects (str and/or IDPropertyGroup)
        #   * IDPropertyGroup (equivalent of dict, with string keys and values of any custom-prop type)
        # IDPropertyArray has a method to_list()
        # IDPropertyGroup has a method to_dict()
        # These types are located in the idprop.types module
        if (value is None) or isinstance(value, bpy.types.ID): return "idblock", 0
        value_type = type(value)
        type_name = cls.type_map.get(value_type)
        if type_name: return type_name, 0
        if value_type is IDPropertyArray: return cls.typecode_map.get(value.typecode), len(value)
        size = (len(value) if value_type is list else 0)
        if string_arrays and (size > 0) and all(isinstance(v, str) for v in value): return "string", size
        return "python", size
    
    @classmethod
    def normalize(cls, value):
        # Convert the value (either a custom prop, or a result of expression) to assignable form
        if value is None: return None
        if isinstance(value, bpy.types.ID): return (value if BlRna.is_valid(value) else None)
        value_type = type(value)
        if value_type in cls.type_map: return value
        if value_type is IDPropertyArray: return value.to_list()
        if value_type is IDPropertyGroup: return value.to_dict()
        if hasattr(value, "items"): return {str(k): cls.normalize(v) for k, v in value.items()}
        return [cls.normalize(v) for v in value if v is not None]
    
    @classmethod
    def _safe_get_metadata(cls, container, key, value):
        is_idblock = isinstance(value, bpy.types.ID)
        
        if is_idblock:
            id_type_info = IDTypes.map("type").get(type(value))
            if id_type_info and cls.id_types_broken.get(id_type_info):
                return None, {"description": "", "default": None, "subtype": 'NONE', "id_type": id_type_info.id_type}
        
        prop_manager = container.id_properties_ui(key)
        metadata = prop_manager.as_dict()
        
        if (is_idblock or (value is None)) and ("id_type" not in metadata):
            if value is None:
                metadata["id_type"] = 'OBJECT'
            else:
                # Note: we can't use a map for this, since cls.id_types
                # may contain base classes, so we have to use isinstance()
                for id_type_info in cls.id_types:
                    if id_type_info.type and isinstance(value, id_type_info.type):
                        metadata["id_type"] = id_type_info.id_type
                        break
        
        metadata.setdefault("description", "") # may not always be present
        
        if isinstance(value, IDPropertyArray):
            # Blender does not ensure that len(default) == len(value)
            default = metadata["default"]
            nv, nd = len(value), len(default)
            if nd < nv:
                metadata["default"] = default + [type(value[0])()] * (nv - nd)
            elif nd > nv:
                metadata["default"] = default[:nv]
        
        return prop_manager, metadata
    
    @classmethod
    def get_metadata(cls, container, key):
        value = container[key]
        if not isinstance(value, cls.types_with_metadata): return None
        
        prop_manager, metadata = cls._safe_get_metadata(container, key, value)
        
        return metadata
    
    @classmethod
    def set_metadata(cls, container, key, metadata, sanitize=False):
        if not metadata: return
        
        value = container[key]
        if not isinstance(value, cls.types_with_metadata): return None
        
        if not sanitize:
            container.id_properties_ui(key).update(**metadata)
            return
        
        prop_manager, sanitized = cls._safe_get_metadata(container, key, value)
        if prop_manager is None: return
        
        for name, dst_value in metadata.items():
            if dst_value is None: continue
            
            if cls.supports_enum and name == "items":
                if isinstance(value, int):
                    dst_value = BpyEnum.normalize_items(dst_value, is_flag=False, separators=False)
                    if dst_value:
                        sanitized[name] = dst_value
                        continue
                
                sanitized.pop(name, None)
                continue
            
            src_value = sanitized.get(name)
            if src_value is None: continue
            
            if name == "subtype":
                type_name, size = cls.get_type_info(value)
                subtype_names = cls.subtype_names.get((type_name, (size > 0)))
                if subtype_names and (dst_value in subtype_names):
                    sanitized[name] = dst_value
                continue
            elif name == "id_type":
                id_type_info = IDTypes.map("id_type").get(dst_value)
                if id_type_info and not cls.id_types_broken.get(id_type_info):
                    sanitized[name] = dst_value
                continue
            
            src_type = type(src_value)
            dst_type = type(dst_value)
            
            converter = cls.converters.get((dst_type, src_type))
            if not converter: continue
            
            dst_value = converter(dst_value, src_value)
            if dst_value is None: continue
            sanitized[name] = dst_value
        
        prop_manager.update(**sanitized)
    
    @classmethod
    def get_overridable(cls, container, key):
        escaped = bpy.utils.escape_identifier(key)
        return container.is_property_overridable_library(f'["{escaped}"]')
    
    @classmethod
    def set_overridable(cls, container, key, is_overridable):
        if is_overridable is None: return
        escaped = bpy.utils.escape_identifier(key)
        container.property_overridable_library_set(f'["{escaped}"]', is_overridable)
    
    @classmethod
    def get_data(cls, container, key):
        return dict(
            value=cls.normalize(container[key]),
            metadata=cls.get_metadata(container, key),
            overridable=cls.get_overridable(container, key),
        )
    
    @classmethod
    def set_data(cls, container, key, data, sanitize=True):
        # Normalize to make sure we don't assign invalid references
        value = data["value"]
        if sanitize: value = cls.normalize(value)
        container[key] = value # assign the value first
        cls.set_metadata(container, key, data.get("metadata"))
        cls.set_overridable(container, key, data.get("overridable"))
    
    @classmethod
    def rename(cls, container, old_key, new_key):
        data = cls.get_data(container, old_key)
        container.pop(old_key, None)
        cls.set_data(container, new_key, data, sanitize=False)
        return data
    
    @classmethod
    def reset(cls, container, key):
        if key not in container: return
        metadata = cls.get_metadata(container, key)
        container[key] = (metadata["default"] if metadata else None)
