#  ***** BEGIN GPL LICENSE BLOCK *****
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#  ***** END GPL LICENSE BLOCK *****

#============================================================================#

'''
Speed tips:
* Static methods are faster than instance/class methods
  Seems like argument passing is a relatively expensive operation
* Closure dictionary lookup with constant key is faster:
    # non-arguments are assumed to be closure-bound objects
    def f(enum): g(enum)
    f(CONST) # something.CONST is even slower
    def f(enum): g(enums[enum])
    f('CONST') # faster
'''

# Convention:
# UPPER_CASE is for enabled/disabled capabilities and constants
# CamelCase is for parameters/functions
# lower_case is for extra functionality

# Alternative:
# NAME/Name for set
# NAME_/Name_ for get
# or: get_Name, set_Name to not conflict with properties

def initialize():
    import math
    from collections import namedtuple
    
    from mathutils import Color, Vector, Matrix, Quaternion, Euler
    
    import numpy as np
    
    import blf
    import gpu
    
    from gpu.types import GPUBatch, GPUIndexBuf, GPUOffScreen, GPUShader, GPUVertBuf, GPUVertFormat
    
    shader_from_builtin = gpu.shader.from_builtin
    
    blf_option_names = ['ROTATION', 'CLIPPING', 'SHADOW', 'KERNING_DEFAULT', 'WORD_WRAP', 'MONOCHROME']
    blf_options = {name : getattr(blf, name) for name in blf_option_names if hasattr(blf, name)}
    
    blf_load = blf.load
    blf_unload = blf.unload
    blf_enable = blf.enable
    blf_disable = blf.disable
    blf_shadow = blf.shadow
    blf_shadow_offset = blf.shadow_offset
    blf_color = blf.color
    blf_position = blf.position
    blf_rotation = blf.rotation
    blf_size = blf.size
    blf_clipping = blf.clipping
    blf_aspect = blf.aspect
    blf_dimensions = blf.dimensions
    blf_draw = blf.draw
    blf_word_wrap = blf.word_wrap
    
    def detect_data_type(buffer):
        try:
            while True:
                value = buffer
                buffer = buffer[0]
        except (TypeError, IndexError):
            pass
        is_float = isinstance(value, float) or np.issubdtype(type(value), np.floating)
        return ('FLOAT' if is_float else 'INT')
    
    def calc_vbo_len(datas):
        vbo_len = -1
        for data in datas:
            if not data: continue
            data_len = len(data)
            if data_len == vbo_len: continue
            if vbo_len >= 0: raise ValueError("Length mismatch for vertex attribute data")
            vbo_len = data_len
        return max(vbo_len, 0)
    
    vbo_format_cache = {}
    
    def recommended_comp_type(attr_type):
        if attr_type in {'FLOAT', 'VEC2', 'VEC3', 'VEC4', 'MAT3', 'MAT4'}:
            return 'F32'
        elif attr_type in {'UINT', 'UVEC2', 'UVEC3', 'UVEC4'}:
            return 'U32'
        else: # attr_type in {'INT', 'IVEC2', 'IVEC3', 'IVEC4', 'BOOL'}
            return 'I32'
    
    def recommended_attr_len(attr_data, attr_name):
        attr_len = 1
        try:
            item = attr_data[attr_name][0]
            while True:
                attr_len *= len(item)
                item = item[0]
        except (TypeError, IndexError):
            pass
        return attr_len
    
    def recommended_fetch_mode(comp_type):
        return ('FLOAT' if comp_type == 'F32' else 'INT')
    
    def shader_format_calc(shader, attr_data):
        # Blender's batch_for_shader() utility method uses
        # custom logic instead of just shader.format_calc()
        vbo_format = GPUVertFormat()
        attrs_info = shader.attrs_info_get()
        for name, attr_type in attrs_info:
            comp_type = recommended_comp_type(attr_type)
            attr_len = recommended_attr_len(attr_data, name)
            fetch_mode = recommended_fetch_mode(comp_type)
            vbo_format.attr_add(id=name, comp_type=comp_type, len=attr_len, fetch_mode=fetch_mode)
        return vbo_format
    
    def to_vbo_format(vbo_format, attr_data):
        if isinstance(vbo_format, GPUVertFormat): return vbo_format
        key = vbo_format
        vbo_format = vbo_format_cache.get(key, None)
        if vbo_format: return vbo_format
        shader = (shader_from_builtin(key) if isinstance(key, str) else key)
        vbo_format = shader_format_calc(shader, attr_data)
        vbo_format_cache[key] = vbo_format
        vbo_format_cache[shader] = vbo_format
        return vbo_format
    
    def convert_line_loop(indices):
        result = [[]]
        
        for count, i in enumerate(indices):
            if count > 1:
                result.append((result[-1][-1], i))
            else:
                result[0].append(i)
        
        result.append((result[-1][-1], result[0][0]))
        
        return GPUIndexBuf(type='LINES', seq=result), 'LINES'
    
    def convert_tri_fan(indices):
        result = [[]]
        
        for count, i in enumerate(indices):
            if count > 2:
                result.append((result[0][0], result[-1][-1], i))
            else:
                result[0].append(i)
        
        return GPUIndexBuf(type='TRIS', seq=result), 'TRIS'
    
    # Since Blender 3.2, LINE_LOOP and TRI_FAN are deprecated
    # (they are not supported in Vulkan / Metal)
    ibo_remappers = {
        'LINE_LOOP': (convert_line_loop, {}),
        'TRI_FAN': (convert_tri_fan, {}),
    }
    
    def to_index_buffer(primitive_type, indices, vbo_len):
        remapper = ibo_remappers.get(primitive_type, None)
        if remapper:
            remapper, cache = remapper
            if indices: return remapper(indices)
            return cache.get(vbo_len, None) or cache.setdefault(vbo_len, remapper(range(vbo_len)))
        elif indices:
            return GPUIndexBuf(type=primitive_type, seq=indices), primitive_type
        else:
            return None, primitive_type
    
    class StateRestorator:
        __slots__ = ("state", "target")
        
        def __init__(self, target, args, kwargs):
            state = {}
            for k in args:
                state[k] = getattr(target, k)
            for k, v in kwargs.items():
                state[k] = getattr(target, k)
                setattr(target, k, v)
            self.state = state
            self.target = target
        
        def restore(self):
            target = self.target
            for k, v in self.state.items():
                setattr(target, k, v)
        
        def __enter__(self):
            return self
        
        def __exit__(self, type, value, traceback):
            self.restore()
    
    class CGL:
        def __call__(self, *args, **kwargs):
            return StateRestorator(self, args, kwargs)
        
        # Adapted from gpu_extras.batch.batch_for_shader()
        @staticmethod
        def batch(*args, **attr_data):
            """
            Positional arguments: vbo format | shader | built-in shader name, primitive type, [indices]
            Keyword arguments: data for the corresponding shader attributes
            """
            
            arg_count = len(args)
            if (arg_count < 2) or (arg_count > 3):
                raise TypeError(f"batch() takes from 2 to 3 positional arguments but {arg_count} were given")
            
            vbo_len = calc_vbo_len(attr_data.values())
            
            vbo_format = to_vbo_format(args[0], attr_data)
            vbo = GPUVertBuf(format=vbo_format, len=vbo_len)
            for id, data in attr_data.items():
                vbo.attr_fill(id=id, data=data)
            
            primitive_type = args[1]
            indices = (args[2] if arg_count > 2 else None)
            ibo, primitive_type = to_index_buffer(primitive_type, indices, vbo_len)
            
            # Blender expects the elem argument to be an actual buffer, if provided
            if not ibo: return GPUBatch(type=primitive_type, buf=vbo)
            return GPUBatch(type=primitive_type, buf=vbo, elem=ibo)
        
        # Note: Blender's gpu read functions return transposed tensors; flatten
        # them via buffer.dimensions = ... or via np.array(buffer).T.reshape(-1)
        
        @staticmethod
        def read_color(xy, wh, channels=4, slot=0, format='UBYTE'):
            "Returns a flattened buffer of colors"
            x, y, w, h = int(xy[0]), int(xy[1]), int(wh[0]), int(wh[1])
            framebuffer = gpu.state.active_framebuffer_get()
            data = framebuffer.read_color(x, y, w, h, channels, slot, format)
            data.dimensions = w * h * channels
            return data
        
        @staticmethod
        def read_depth(xy, wh):
            "Returns a flattened buffer of depth values"
            x, y, w, h = int(xy[0]), int(xy[1]), int(wh[0]), int(wh[1])
            framebuffer = gpu.state.active_framebuffer_get()
            data = framebuffer.read_depth(x, y, w, h)
            data.dimensions = w * h
            return data
        
        @staticmethod
        def read_zbuffer(xy, wh=(1, 1), centered=False, src=None):
            "Returns a (H, W) buffer of depth values"
            x, y, w, h = int(xy[0]), int(xy[1]), int(wh[0]), int(wh[1])
            
            if isinstance(wh, (int, float)):
                wh = (wh, wh)
            elif len(wh) < 2:
                wh = (wh[0], wh[0])
            
            if centered:
                x -= w // 2
                y -= h // 2
            
            # Note: xy is in window coordinates
            if src is None:
                framebuffer = gpu.state.active_framebuffer_get()
                zbuf = framebuffer.read_depth(x, y, w, h)
            else:
                src, w0, h0 = src
                template = np.zeros((h, w))
                for dy in range(h):
                    y0 = min(max(y + dy, 0), h0-1)
                    for dx in range(w):
                        x0 = min(max(x + dx, 0), w0-1)
                        template[dy, dx] = src[y0][x0]
                zbuf = gpu.types.Buffer('FLOAT', (h, w), template)
            
            return zbuf
        
        @staticmethod
        def rgba8_texture(size, buffer):
            # Note: buffer is expected to store RGBA8 data (typecally as 4-channel byte or 1-channel int)
            # gpu.types.Buffer requires numpy arrays to be C-contiguous
            pixels = np.array(buffer, order='C').view("uint8")
            pixels = np.array(pixels / 255.0, dtype='float32', order='C')
            pixels.shape = (size[1], size[0], 4)
            # For now, only FLOAT buffers are supported (maybe this would change in the future)
            buffer = gpu.types.Buffer('FLOAT', pixels.shape, pixels)
            return gpu.types.GPUTexture(size, data=buffer)
        
        @classmethod
        def rgba_texture(cls, size, buffer):
            if detect_data_type(buffer) == 'INT':
                return cls.rgba8_texture(size, buffer)
            
            return gpu.types.GPUTexture(size, data=buffer)
        
        @classmethod
        def rgba8_to_buffer(cls, size, pixels):
            buffer = gpu.types.Buffer('INT', size[0]*size[1], pixels)
            
            # For some reason, on MacOS buffer may contain the same values, but as floats
            # (perhaps MacOS version of blender only supports float buffers?)
            if isinstance(buffer[0], float):
                # This is much faster than using preview.image_pixels_float
                pixels = np.array(pixels, dtype='int32', order='C').view("uint8")
                pixels = np.array(pixels / 255.0, dtype='float32', order='C')
                pixels.shape = (size[1], size[0], 4)
                buffer = gpu.types.Buffer('FLOAT', pixels.shape, pixels)
            
            return buffer
        
        @classmethod
        def preview_to_buffer(cls, preview):
            return cls.rgba8_to_buffer(preview.image_size, preview.image_pixels[:])
    
    cgl = CGL()
    
    # ========== TEXT ========== #
    
    class TextWrapper:
        # dimensions & wrapping calculation
        @classmethod
        def dimensions(cls, text, font=0):
            return blf_dimensions(font, text)
        
        @classmethod
        def _split_word(cls, width, x, max_x, word, lines, font):
            line = ""
            
            for c in word:
                x_dx = x + blf_dimensions(font, line+c)[0]
                
                if x_dx > width:
                    x_dx = x + blf_dimensions(font, line)[0]
                    lines.append(line)
                    line = c
                    x = 0
                else:
                    line += c
                
                max_x = max(x_dx, max_x)
            
            return line, x, max_x

        @classmethod
        def _split_line(cls, width, x, max_x, line, lines, font):
            words = line.split(" ")
            line = ""
            
            for word in words:
                c = (word if not line else " " + word)
                x_dx = x + blf_dimensions(font, line+c)[0]
                
                if x_dx > width:
                    x_dx = x + blf_dimensions(font, line)[0]
                    if not line:
                        line, x, max_x = cls._split_word(width, x, max_x, word, lines, font)
                    else:
                        lines.append(line)
                        line, x, max_x = cls._split_word(width, 0, max_x, word, lines, font)
                    x = 0
                else:
                    line += c
                
                max_x = max(x_dx, max_x)
            
            if line: lines.append(line)
            
            return max_x

        @classmethod
        def split_text(cls, width, x, max_x, text, lines, font=0):
            if width is None: width = math.inf
            
            for line in text.splitlines():
                if not line:
                    lines.append("")
                else:
                    max_x = cls._split_line(width, x, max_x, line, lines, font)
                x = 0
            
            return max_x

        @classmethod
        def wrap_text(cls, text, width, indent=0, font=0):
            """
            Splits text into lines that don't exceed the given width.
            text -- the text
            width -- the width the text should fit into
            font -- the id of the typeface as returned by blf.load(). Defaults to 0 (the default font)
            indent -- the indent of the paragraphs. Defaults to 0
            Returns: lines, size
            lines -- the list of the resulting lines
            size -- actual (width, height) of these lines
            """
            
            if width is None: width = math.inf
            
            lines = []
            max_x = 0
            for line in text.splitlines():
                if not line:
                    lines.append("")
                else:
                    max_x = cls._split_line(width, indent, max_x, line, lines, font)
            
            line_height = blf_dimensions(font, "Ig")[1]
            
            return lines, (max_x, len(lines)*line_height)
    
    class BatchedText:
        __slots__ = ("font", "pieces", "size")
        
        def __init__(self, font, pieces, size):
            self.font = font
            self.pieces = pieces
            self.size = size
        
        def draw(self, pos, origin=None):
            x = pos[0]
            y = pos[1]
            z = (pos[2] if len(pos) > 2 else 0)
            
            if origin:
                x -= self.size[0] * origin[0]
                y -= self.size[1] * origin[1]
            
            # TODO: adjust this code when antialiasing control
            # gets implemented in the gpu module
            prev_blend = cgl.blend
            # prev_polygon_smooth = cgl.POLYGON_SMOOTH
            
            cgl.blend = 'ALPHA'
            # cgl.POLYGON_SMOOTH = False
            
            font = self.font
            x0, y0 = round(x), round(y)
            for txt, x, y in self.pieces:
                blf_position(font, x0+x, y0+y, z)
                blf_draw(font, txt)
            
            # Note: blf_draw() resets GL_BLEND (and GL_POLYGON_SMOOTH
            # since Blender 2.91), so we have to restore them anyway
            
            # cgl.POLYGON_SMOOTH = prev_polygon_smooth
            cgl.blend = prev_blend
    
    class Text:
        font = 0 # 0 is the default font
        
        # load / unload
        def load(self, filename, size=None):
            font = blf_load(filename)
            if size is not None: blf_size(font, int(size))
            return font
        def unload(self, filename):
            blf_unload(filename)
        
        # enable / disable options
        def enable(self, option):
            if option not in blf_options: return
            blf_enable(self.font, blf_options[option])
        def disable(self, option):
            if option not in blf_options: return
            blf_disable(self.font, blf_options[option])
        
        # set shadow
        def shadow(self, level, r, g, b, a):
            blf_shadow(self.font, level, r, g, b, a)
        def shadow_offset(self, x, y):
            blf_shadow_offset(self.font, x, y)
        
        # set position / rotation / size / color
        def position(self, x, y, z=0.0):
            blf_position(self.font, x, y, z)
        def rotation(self, angle):
            blf_rotation(self.font, angle)
        def size(self, size):
            blf_size(self.font, int(size))
        def color(self, r, g, b, a=1.0):
            blf_color(self.font, r, g, b, a)
        
        # set clipping / aspect
        def clipping(self, xmin, ymin, xmax, ymax):
            blf_clipping(self.font, xmin, ymin, xmax, ymax)
        def aspect(self, aspect):
            blf_aspect(self.font, aspect)
        
        def compile(self, text, width=None, alignment=None, spacing=1.0):
            font = self.font
            
            line_height = blf_dimensions(font, "Ig")[1]
            topline = blf_dimensions(font, "I")[1]
            
            lines, size = TextWrapper.wrap_text(text, width, font=font)
            
            w, h = size[0], size[1] * abs(spacing)
            
            size = (w, h)
            pieces = []
            
            if (alignment in (None, 'LEFT')): alignment = 0.0
            elif (alignment == 'CENTER'): alignment = 0.5
            elif (alignment == 'RIGHT'): alignment = 1.0
            
            # blf text origin is at lower left corner, and +Y is "up"
            # But since text is usually read from top to bottom,
            # consider positive spacing to be "down".
            if spacing > 0: lines = reversed(lines)
            
            y_step = line_height * abs(spacing)
            
            x, y = 0, 0
            for line in lines:
                x = (w - blf_dimensions(font, line)[0]) * alignment
                pieces.append((line, round(x), round(y)))
                y += y_step
            
            return BatchedText(font, pieces, size)
        
        def draw(self, text, pos=None, origin=None, width=None, alignment=None, spacing=1.0):
            if pos is None:
                # if position is not specified, other calculations cannot be performed
                blf_draw(self.font, text)
                return None
            
            batched = self.compile(text, width, alignment, spacing)
            batched.draw(pos, origin)
            return batched
    
    cgl.text = Text()
    
    # ========== GPU API ========== #
    
    def add_descriptor(name, getter, setter, doc=""):
        #Descriptor = type(name+"_Descriptor", (), {"__doc__":doc, "__get__":getter, "__set__":setter})
        class Descriptor:
            __doc__ = doc
            __get__ = getter
            __set__ = setter
        setattr(CGL, name, Descriptor())
    
    def _get(self, instance, owner):
        return gpu.state.blend_get()
    def _set(self, instance, value):
        gpu.state.blend_set(value)
    add_descriptor("blend", _get, _set)
    
    def _get(self, instance, owner):
        return None
    def _set(self, instance, value):
        gpu.state.clip_distances_set(int(value))
    add_descriptor("clip_distances", _get, _set)
    
    def _get(self, instance, owner):
        return None
    def _set(self, instance, value):
        gpu.state.color_mask_set(bool(value[0]), bool(value[1]), bool(value[2]), bool(value[3]))
    add_descriptor("color_mask", _get, _set)
    
    def _get(self, instance, owner):
        return gpu.state.depth_mask_get()
    def _set(self, instance, value):
        gpu.state.depth_mask_set(value)
    add_descriptor("depth_mask", _get, _set)
    
    def _get(self, instance, owner):
        return gpu.state.depth_test_get()
    def _set(self, instance, value):
        gpu.state.depth_test_set(value)
    add_descriptor("depth_test", _get, _set)
    
    def _get(self, instance, owner):
        return None
    def _set(self, instance, value):
        gpu.state.face_culling_set(value)
    add_descriptor("face_culling", _get, _set)
    
    def _get(self, instance, owner):
        return None
    def _set(self, instance, value):
        gpu.state.front_facing_set(value)
    add_descriptor("front_facing", _get, _set)
    
    def _get(self, instance, owner):
        return gpu.state.line_width_get()
    def _set(self, instance, value):
        gpu.state.line_width_set(value)
    add_descriptor("line_width", _get, _set)
    
    def _get(self, instance, owner):
        return None
    def _set(self, instance, value):
        gpu.state.point_size_set(value)
    add_descriptor("point_size", _get, _set)
    
    def _get(self, instance, owner):
        return None
    def _set(self, instance, value):
        gpu.state.program_point_size_set(value)
    add_descriptor("program_point_size", _get, _set)
    
    def _get(self, instance, owner):
        return gpu.state.scissor_get()
    def _set(self, instance, value):
        gpu.state.scissor_test_set(bool(value))
        if value: gpu.state.scissor_set(int(value[0]), int(value[1]), int(value[2]), int(value[3]))
    add_descriptor("scissor", _get, _set)
    
    def _get(self, instance, owner):
        return gpu.state.viewport_get()
    def _set(self, instance, value):
        gpu.state.viewport_set(int(value[0]), int(value[1]), int(value[2]), int(value[3]))
    add_descriptor("viewport", _get, _set)
    
    return {"cgl":cgl, "TextWrapper":TextWrapper}

globals().update(initialize())
del initialize
