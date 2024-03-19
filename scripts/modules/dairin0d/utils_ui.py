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

import re
import pickle
import base64

import bpy
import blf

from mathutils import Color, Vector, Matrix, Quaternion, Euler

from .utils_python import DummyObject
from .bounds import Bounds
from .utils_gl import TextWrapper
from .bpy_inspect import BlRna, prop, IDTypes, BpyEnum

#============================================================================#

# Note: making a similar wrapper for Operator.report is impossible,
# since Blender only shows the report from the currently executing operator.

# ===== MESSAGEBOX ===== #
if not hasattr(bpy.types, "WM_OT_messagebox"):
    class WM_OT_messagebox(bpy.types.Operator):
        bl_idname = "wm.messagebox"
        
        # "Attention!" is quite generic caption that suits
        # most of the situations when "OK" button is desirable.
        # bl_label isn't really changeable at runtime
        # (changing it causes some memory errors)
        bl_label = "Attention!"
        
        # We can't pass arguments through normal means,
        # since in this case a "Reset" button would appear
        args = {}
        
        # If we don't define execute(), there would be
        # an additional label "*Redo unsupported*"
        def execute(self, context):
            return {'FINISHED'}
        
        def invoke(self, context, event):
            text = self.args.get("text", "")
            self.icon = self.args.get("icon", 'NONE')
            if (not text) and (self.icon == 'NONE'):
                return {'CANCELLED'}
            
            border_w = 8*2
            icon_w = (0 if (self.icon == 'NONE') else 16)
            w_incr = border_w + icon_w
            
            width = self.args.get("width", 300) - border_w
            
            self.lines = []
            max_x = TextWrapper.split_text(width, icon_w, 0, text, self.lines, font=0)
            width = max_x + border_w
            
            self.spacing = self.args.get("spacing", 0.5)
            self.spacing = max(self.spacing, 0.0)
            
            wm = context.window_manager
            
            confirm = self.args.get("confirm", False)
            
            if confirm:
                return wm.invoke_props_dialog(self, width=int(width))
            else:
                return wm.invoke_popup(self, width=int(width))
        
        def draw(self, context):
            layout = self.layout
            
            col = layout.column()
            col.scale_y = 0.5 * (1.0 + self.spacing * 0.5)
            
            icon = self.icon
            for line in self.lines:
                if icon != 'NONE': line = " "+line
                col.label(text=line, icon=icon)
                icon = 'NONE'
    
    bpy.utils.register_class(WM_OT_messagebox) # REGISTER

def messagebox(text, icon='NONE', width=300, confirm=False, spacing=0.5):
    """
    Displays a message box with the given text and icon.
    text -- the messagebox's text
    icon -- the icon (displayed at the start of the text)
        Defaults to 'NONE' (no icon).
    width -- the messagebox's max width
        Defaults to 300 pixels.
    confirm -- whether to display "OK" button (this is purely
        cosmetical, as the message box is non-blocking).
        Defaults to False.
    spacing -- relative distance between the lines
        Defaults to 0.5.
    """
    WM_OT_messagebox = bpy.types.WM_OT_messagebox
    WM_OT_messagebox.args["text"] = text
    WM_OT_messagebox.args["icon"] = icon
    WM_OT_messagebox.args["width"] = width
    WM_OT_messagebox.args["spacing"] = spacing
    WM_OT_messagebox.args["confirm"] = confirm
    bpy.ops.wm.messagebox('INVOKE_DEFAULT')

#============================================================================#

# ===== HIERARCHY ENUM ===== #
if not hasattr(bpy.types, "WM_PT_hierarchy_enum_menu"):
    class EnumParsingNode:
        def __init__(self):
            self.dict = {}
            self.items = []
        
        def get(self, key):
            node = self.dict.get(key)
            if not node:
                node = EnumParsingNode()
                self.dict[key] = node
            return node
    
    class WM_PT_hierarchy_enum_menu(bpy.types.Panel):
        bl_label = ""
        bl_space_type = 'CONSOLE'
        bl_region_type = 'WINDOW'
        bl_options = {'DEFAULT_CLOSED', 'INSTANCED'}
        
        args_info = {}
        
        @classmethod
        def parse_enum(cls, enum_items, separator, source_id=0, exclude=()):
            if isinstance(separator, str):
                splitter = (lambda s: s.split(separator))
            elif isinstance(separator, re.Pattern):
                splitter = (lambda s: separator.split(s))
            elif hasattr(separator, "__iter__"):
                separator = re.compile("|".join(f"(?:{re.escape(v)})" for v in separator))
                splitter = (lambda s: separator.split(s))
            else:
                splitter = separator
            
            root = EnumParsingNode()
            
            for item in enum_items:
                if not item: continue # separator
                if item.identifier in exclude: continue
                
                parts = splitter(item[source_id])
                idname = item.identifier
                label = (item.name if source_id != 1 else parts[-1])
                
                node = root
                for part in parts:
                    node = node.get(part)
                
                node.items.append((idname, label))
            
            return root
        
        @classmethod
        def encode(cls, data):
            return base64.b64encode(pickle.dumps(data)).decode("utf-8")
        
        @classmethod
        def decode(cls, value):
            return pickle.loads(base64.b64decode(value.encode("utf-8")))
        
        @classmethod
        def draw_main(cls, layout, data, property, separator, source_id=0, exclude=(), override={}, **kwargs):
            # draw_main can only be invoked when the menu sub-items are not displayed
            args_info = cls.args_info
            args_info.clear()
            
            value = getattr(data, property)
            label = layout.enum_item_name(data, property, value)
            icon_value = layout.enum_item_icon(data, property, value)
            
            info = (separator, source_id, exclude, override, kwargs.pop("width", None), kwargs.pop("items", None))
            info_str = cls.encode(info)
            
            row = layout.row(align=True)
            row.context_pointer_set("@data", data)
            BlUI.set_ui_metadata(row, {"property":property, "info":info_str}) # do this before popover()
            row.popover("WM_PT_hierarchy_enum_menu", text=label, icon_value=icon_value, translate=False)
        
        def draw(self, context):
            data = getattr(context, "@data", None)
            if data is None: return
            
            metadata = BlUI.get_ui_metadata(context)
            property = metadata.get("property")
            if property is None: return
            
            args_key = (data, property)
            
            # Here we exploit the fact that Blender can only display one branch of menu tree at a time
            cls = self.__class__
            args_info = cls.args_info
            
            if args_info.get("key") != args_key:
                args_info.clear()
                
                info_str = metadata.get("info")
                if not info_str: return
                
                info = cls.decode(info_str)
                separator, source_id, exclude, override, width, items_fallback = info
                
                # Blender currently does not provide an ability to get dynamically generated
                # enum items (or the operator's python class) via RNA API, so here's a hack
                # Note: items_fallback isn't a callback because e.g. local functions can't be pickled
                if items_fallback is not None:
                    enum_items = (BpyEnum.normalize_item(item, wrap=True) for item in items_fallback)
                else:
                    enum_items = BlRna.enum_items(data, property, container=iter)
                
                args_info["key"] = args_key
                args_info["root"] = cls.parse_enum(enum_items, separator, source_id, exclude)
                args_info["override"] = override
                args_info["width"] = width
            
            root = args_info["root"]
            override = args_info["override"]
            width = args_info["width"]
            
            layout = self.layout
            # layout.emboss = 'NONE_OR_STATUS'
            layout.emboss = 'PULLDOWN_MENU'
            
            col = layout.column(align=True)
            
            labels = []
            
            def draw_entry(idname, label, sub_metadata):
                labels.append(label)
                
                row = col.row(align=True)
                row.alignment = 'EXPAND'
                subrow = row.row(align=False)
                if idname is None:
                    subrow.label(text=label)
                else:
                    if getattr(data, property) == idname:
                        subrow.emboss = 'NORMAL'
                    subrow.prop_enum(data, property, idname, text=label)
                
                subrow = row.row(align=True)
                subrow.alignment = 'RIGHT'
                subrow.enabled = bool(sub_metadata)
                if sub_metadata:
                    BlUI.set_ui_metadata(subrow, sub_metadata) # do this before popover()
                    subrow.popover("WM_PT_hierarchy_enum_menu", text="", icon='DISCLOSURE_TRI_RIGHT')
                else:
                    subrow.label(text="", icon='BLANK1')
            
            path = metadata.get("path")
            path = (cls.decode(path) if path else [])
            
            node = root
            for key in path:
                node = node.dict[key]
            
            for key, subnode in node.dict.items():
                if subnode.dict:
                    sub_metadata = {"path": cls.encode(path+[key])}
                else:
                    sub_metadata = None
                
                if subnode.items:
                    for idname, label in subnode.items:
                        draw_entry(idname, label, sub_metadata)
                else:
                    draw_entry(None, key, sub_metadata)
            
            max_x = 0
            for label in labels:
                max_x = max(max_x, TextWrapper.dimensions(label)[0])
            max_x += BlUI.ICON_SIZE * 2
            if width and not path: max_x = max(max_x, width)
            
            layout.ui_units_x = (max_x / BlUI.ICON_SIZE)
    
    bpy.utils.register_class(WM_PT_hierarchy_enum_menu) # REGISTER

#============================================================================#

# Note:
# if item is property group instance and item["pi"] = 3.14,
# in UI it should be displayed like this: layout.prop(item, '["pi"]')

# ===== NESTED LAYOUT ===== #
class NestedLayout:
    """
    Utility for writing more structured UI drawing code.
    Attention: layout properties are propagated to sublayouts!
    
    Example:
    
    def draw(self, context):
        layout = NestedLayout(self.layout, self.bl_idname)
        
        exit_layout = True
        
        # You can use both the standard way:
        
        sublayout = layout.split()
        sublayout.label("label A")
        sublayout.label("label B")
        
        # And the structured way:
        
        with layout:
            layout.label("label 1")
            if exit_layout: layout.exit()
            layout.label("label 2") # won't be executed
        
        with layout.row(True)["main"]:
            layout.label("label 3")
            with layout.row(True)(enabled=False):
                layout.label("label 4")
                if exit_layout: layout.exit("main")
                layout.label("label 5") # won't be executed
            layout.label("label 6") # won't be executed
        
        with layout.fold("Foldable micro-panel", "box"):
            if layout.folded: layout.exit()
            layout.label("label 7")
            with layout.fold("Foldable 2"):
                layout.label("label 8") # not drawn if folded
    """
    
    _sub_names = {"row", "column", "column_flow", "grid_flow", "box", "split", "menu_pie"}
    
    _default_attrs = dict(
        active = True,
        alert = False,
        alignment = 'EXPAND',
        enabled =  True,
        operator_context = 'INVOKE_DEFAULT',
        scale_x = 1.0,
        scale_y = 1.0,
    )
    
    def __new__(cls, layout, idname="", parent=None):
        """
        Wrap the layout in a NestedLayout.
        To avoid interference with other panels' foldable
        containers, supply panel's bl_idname as the idname.
        """
        if isinstance(layout, cls) and (layout._idname == idname): return layout
        
        self = object.__new__(cls)
        self._idname = idname
        self._parent = parent
        self._layout = layout
        self._stack = [self]
        self._attrs = dict(self._default_attrs)
        self._tag = None
        
        # propagate settings to sublayouts
        if parent: self(**parent._stack[-1]._attrs)
        
        return self
    
    def __len__(self):
        return len(self._stack)
    
    def __getitem__(self, index):
        return self._stack[index]
    
    def __getattr__(self, name):
        layout = self._stack[-1]._layout
        if not layout:
            # This is the dummy layout; imitate normal layout
            # behavior without actually drawing anything.
            if name in self._sub_names:
                return (lambda *args, **kwargs: NestedLayout(None, self._idname, self))
            else:
                return self._attrs.get(name, self._dummy_callable)
        
        if name in self._sub_names:
            func = getattr(layout, name)
            return (lambda *args, **kwargs: NestedLayout(func(*args, **kwargs), self._idname, self))
        else:
            return getattr(layout, name)
    
    def __setattr__(self, name, value):
        if name.startswith("_"):
            self.__dict__[name] = value
        else:
            wrapper = self._stack[-1]
            wrapper._attrs[name] = value
            if wrapper._layout: setattr(wrapper._layout, name, value)
    
    def __call__(self, **kwargs):
        """Batch-set layout attributes."""
        wrapper = self._stack[-1]
        wrapper._attrs.update(kwargs)
        layout = wrapper._layout
        if layout:
            for k, v in kwargs.items():
                setattr(layout, k, v)
        return self
    
    @staticmethod
    def _dummy_callable(*args, **kwargs):
        return NestedLayout._dummy_obj
    _dummy_obj = DummyObject()
    
    def prop_enum_filtered(self, data, property, exclude=(), override={}, text_ctxt="", translate=True):
        BlUI.prop_enum_filtered(self, data, property, exclude, override, text_ctxt, translate)
    
    def prop_menu_enum_hierarchy(self, data, property, separator, source_id=0, exclude=(), override={}, **kwargs):
        BlUI.prop_menu_enum_hierarchy(self, data, property, separator, source_id, exclude, override, **kwargs)
    
    def custom_any_ID(self, data, property, type_property, text=None, text_ctxt="", translate=True):
        BlUI.custom_any_ID(self, data, property, type_property, text, text_ctxt, translate)
    
    # ===== FOLD (currently very hacky) ===== #
    # Each foldable micropanel needs to store its fold-status
    # as a Bool property (in order to be clickable in the UI)
    # somewhere where it would be saved with .blend, but won't
    # be affected by most of the other things (i.e., in Screen).
    # At first I thought to implement such storage with
    # nested dictionaries, but currently layout.prop() does
    # not recognize ID-property dictionaries as a valid input.
    class FoldPG(bpy.types.PropertyGroup):
        def update(self, context):
            pass # just indicates that the widget needs to be force-updated
        value: False | prop("", "Fold/unfold", update=update)
    bpy.utils.register_class(FoldPG) # REGISTER
    
    # make up some name that's unlikely to be used by normal addons
    folds_keyname = "dairin0d_ui_utils_NestedLayout_ui_folds"
    setattr(bpy.types.Screen, folds_keyname, [FoldPG] | prop()) # REGISTER
    
    folded = False # stores folded status from the latest fold() call
    
    def fold(self, text, container=None, folded=False, key=None, text_ctxt="", translate=True):
        """
        Create a foldable container.
        text -- the container's title/label
        container -- a sequence (type_of_container, arg1, ..., argN)
            where type_of_container is one of {"row", "column",
            "column_flow", "box", "split"}; arg1..argN are the
            arguments of the corresponding container function.
            If you supply just the type_of_container, it would be
            interpreted as (type_of_container,).
        folded -- whether the container should be folded by default.
            Default value is False.
        key -- the container's unique identifier within the panel.
            If not specified, the container's title will be used
            in its place.
        """
        data_path = "%s:%s" % (self._idname, key or text)
        folds = getattr(bpy.context.screen, self.folds_keyname)
        
        try:
            this_fold = folds[data_path]
        except KeyError:
            this_fold = folds.add()
            this_fold.name = data_path
            this_fold.value = folded
        
        is_fold = this_fold.value
        icon = ('DOWNARROW_HLT' if not is_fold else 'RIGHTARROW')
        
        # make the necessary container...
        if not container:
            container_args = ()
            container = "column"
        elif isinstance(container, str):
            container_args = ()
        else:
            container_args = container[1:]
            container = container[0]
        res = getattr(self, container)(*container_args)
        
        with res.row(True)(alignment='LEFT'):
            res.prop(this_fold, "value", text=text, icon=icon, emboss=False,
                toggle=True, text_ctxt=text_ctxt, translate=translate)
        
        # make fold-status accessible to the calling code
        self.__dict__["folded"] = is_fold
        
        # If folded, return dummy layout
        if is_fold: return NestedLayout(None, self._idname, self)
        
        return res
    
    # ===== NESTED CONTEXT MANAGEMENT ===== #
    class ExitSublayout(Exception):
        def __init__(self, tag=None):
            self.tag = tag
    
    @classmethod
    def exit(cls, tag=None):
        """Jump out of current (or marked with the given tag) layout's context."""
        raise cls.ExitSublayout(tag)
    
    def __getitem__(self, tag):
        """Mark this layout with the tag"""
        self._tag = tag
        return self
    
    def __enter__(self):
        # Only nested (context-managed) layouts are stored in stack
        parent = self._parent
        if parent: parent._stack.append(self)
        return self
    
    def __exit__(self, type, value, traceback):
        # Only nested (context-managed) layouts are stored in stack
        parent = self._parent
        if parent: parent._stack.pop()
        
        if type == self.ExitSublayout:
            # Is this the layout the exit() was requested for?
            # Yes: suppress the exception. No: let it propagate to the parent.
            return (value.tag is None) or (value.tag == self._tag)

#============================================================================#

class BlUI:
    # These constants (and ui_scale, region_scale, actual_region_width)
    # are taken from the addon "development_icon_get.py"
    DPI = 72
    POPUP_PADDING = 10
    PANEL_PADDING = 44
    WIN_PADDING = 32
    ICON_SIZE = 20
    
    def ui_scale(context=None):
        if context is None: context = bpy.context
        prefs = context.preferences.system
        return prefs.dpi / BlUI.DPI
    
    def region_scale(region=None):
        if region is None: region = bpy.context.region
        _, y0 = region.view2d.region_to_view(0, 0)
        _, y1 = region.view2d.region_to_view(0, 10)
        return 10 / max(0.01, abs(y0 - y1))
    
    def actual_region_width(region=None):
        if region is None: region = bpy.context.region
        return region.width - BlUI.PANEL_PADDING * BlUI.ui_scale()
    
    def actual_window_width(window=None):
        if window is None: window = bpy.context.window
        return window.width - BlUI.WIN_PADDING * BlUI.ui_scale()
    
    def tag_redraw(arg=None):
        """A utility function to tag redraw of arbitrary UI units."""
        if arg is None:
            arg = bpy.context.window_manager
        elif isinstance(arg, bpy.types.Window):
            arg = arg.screen
        
        if isinstance(arg, bpy.types.Screen):
            for area in arg.areas:
                area.tag_redraw()
        elif isinstance(arg, bpy.types.WindowManager):
            for window in arg.windows:
                for area in window.screen.areas:
                    area.tag_redraw()
        else: # Region, Area, RenderEngine
            arg.tag_redraw()
    
    def calc_region_rect(area, r, overlap=True, convert=None):
        # Note: there may be more than one region of the same type (e.g. in quadview)
        if (not overlap) and (r.type == 'WINDOW'):
            x0, y0, x1, y1 = r.x, r.y, r.x+r.width, r.y+r.height
            
            for r in area.regions:
                if (r.width <= 0) or (r.height <= 0): continue
                
                # A HUD-specific hack. At least in Blender 2.80, HUD in 3d view
                # in some cases does not become 1x1 when it's "hidden",
                # but we may still attempt to detect it by its (x,y) being zero
                if (r.alignment == 'FLOAT') and (r.x == 0) and (r.y == 0): continue
                
                alignment = r.alignment
                if convert: alignment = convert.get(alignment, alignment)
                
                if alignment == 'TOP':
                    y1 = min(y1, r.y)
                elif alignment == 'BOTTOM':
                    y0 = max(y0, r.y + r.height)
                elif alignment == 'LEFT':
                    x0 = max(x0, r.x + r.width)
                elif alignment == 'RIGHT':
                    x1 = min(x1, r.x)
            
            return Bounds.MinSize((x0, y0), (x1-x0, y1-y0))
        else:
            return Bounds.MinSize((r.x, r.y), (r.width, r.height))
    
    def point_in_rect(p, r):
        return ((p[0] >= r.x) and (p[0] < r.x + r.width) and (p[1] >= r.y) and (p[1] < r.y + r.height))
    
    def rv3d_from_region(area, region):
        if (area.type != 'VIEW_3D') or (region.type != 'WINDOW'): return None
        
        space_data = area.spaces.active
        try:
            quadviews = space_data.region_quadviews
        except AttributeError:
            quadviews = None # old API
        
        if not quadviews: return space_data.region_3d
        
        x_id = 0
        y_id = 0
        for r in area.regions:
            if (r.type == 'WINDOW') and (r != region):
                if r.x < region.x: x_id = 1
                if r.y < region.y: y_id = 1
        
        # 0: bottom left (Front Ortho)
        # 1: top left (Top Ortho)
        # 2: bottom right (Right Ortho)
        # 3: top right (User Persp)
        return quadviews[y_id | (x_id << 1)]
    
    # areas can't overlap, but regions can
    def ui_contexts_under_coord(x, y, window=None):
        point = int(x), int(y)
        
        if not window: window = bpy.context.window
        
        screen = window.screen
        scene = screen.scene
        tool_settings = scene.tool_settings
        
        for area in screen.areas:
            if not BlUI.point_in_rect(point, area): continue
            
            space_data = area.spaces.active
            for region in area.regions:
                if BlUI.point_in_rect(point, region):
                    yield dict(window=window, screen=screen,
                        area=area, space_data=space_data, region=region,
                        region_data=BlUI.rv3d_from_region(area, region),
                        scene=scene, tool_settings=tool_settings)
            break
    
    def ui_context_under_coord(x, y, index=0, window=None):
        ui_context = None
        for i, ui_context in enumerate(BlUI.ui_contexts_under_coord(x, y, window)):
            if i == index: return ui_context
        return ui_context
    
    def find_ui_area(area_type, region_type='WINDOW', window=None):
        if not window: window = bpy.context.window
        
        compare_directly = isinstance(area_type, bpy.types.Area)
        
        screen = window.screen
        scene = window.scene
        tool_settings = scene.tool_settings
        
        for area in screen.areas:
            if compare_directly:
                if area != area_type: continue
            else:
                if area.type != area_type: continue
            
            space_data = area.spaces.active
            region = None
            
            for _region in area.regions:
                if _region.type == region_type: region = _region
            
            return dict(window=window, screen=screen,
                area=area, space_data=space_data, region=region,
                region_data=BlUI.rv3d_from_region(area, region),
                scene=scene, tool_settings=tool_settings)
    
    def ui_hierarchy(ui_obj):
        if isinstance(ui_obj, bpy.types.Window):
            return (ui_obj, None, None)
        elif isinstance(ui_obj, bpy.types.Area):
            wm = bpy.context.window_manager
            for window in wm.windows:
                for area in window.screen.areas:
                    if area == ui_obj: return (window, area, None)
        elif isinstance(ui_obj, bpy.types.Region):
            wm = bpy.context.window_manager
            for window in wm.windows:
                for area in window.screen.areas:
                    for region in area.regions:
                        if region == ui_obj: return (window, area, region)
    
    # TODO: relative coords?
    def convert_ui_coord(area, region, xy, src, dst, vector=True):
        x, y = xy
        if src == dst:
            pass
        elif src == 'WINDOW':
            if dst == 'AREA':
                x -= area.x
                y -= area.y
            elif dst == 'REGION':
                x -= region.x
                y -= region.y
        elif src == 'AREA':
            if dst == 'WINDOW':
                x += area.x
                y += area.y
            elif dst == 'REGION':
                x += area.x - region.x
                y += area.y - region.y
        elif src == 'REGION':
            if dst == 'WINDOW':
                x += region.x
                y += region.y
            elif dst == 'AREA':
                x += region.x - area.x
                y += region.y - area.y
        return (Vector((x, y)) if vector else (int(x), int(y)))
    
    _ui_metadata_counter = 0
    
    @classmethod
    def set_ui_metadata(cls, layout, metadata, prefix="$"):
        cls._ui_metadata_counter += 1
        version = f"{cls._ui_metadata_counter:0>16x}" # 16-char 0-padded hex value
        
        for key, value in metadata.items():
            entry = (f"{key}:{version}" if value is None else f"{key}:{version}={value}")
            layout.context_pointer_set(prefix+entry, None)
    
    @classmethod
    def get_ui_metadata(cls, context, prefix="$"):
        # Since dir() returns sorted values, we can ensure that only the latest
        # assigned values are returned by including a version/counter in the key
        
        start = len(prefix)
        metadata = {}
        for key in dir(context):
            if not key.startswith(prefix): continue
            
            i = key.find("=", start)
            if i < 0:
                value = None
                key = key[start:]
            else:
                value = key[i+1:]
                key = key[start:i]
            
            i = key.rfind(":")
            if i >= 0: key = key[:i]
            
            metadata[key] = value
        
        return metadata
    
    # Utility UI drawing methods:
    
    def prop_enum_filtered(layout, data, property, exclude=(), override={}, text_ctxt="", translate=True):
        if override is None:
            override = (lambda identifier: {})
        elif isinstance(override, dict):
            kwargs = dict(override)
            override = (lambda identifier: kwargs)
        
        for item in BlRna.enum_items(data, property, container=iter):
            if item is None:
                if item not in exclude: layout.separator()
                continue
            
            if item.identifier in exclude: continue
            
            kwargs = override(item.identifier)
            kwargs.setdefault("text_ctxt", text_ctxt)
            kwargs.setdefault("translate", translate)
            layout.prop_enum(data, property, item.identifier, **kwargs)
    
    def prop_menu_enum_hierarchy(layout, data, property, separator, source_id=0, exclude=(), override={}, **kwargs):
        draw_main = bpy.types.WM_PT_hierarchy_enum_menu.draw_main # shared class, so use bpy.types
        draw_main(layout, data, property, separator, source_id, exclude, override, **kwargs)
    
    # Blender's built-in  prop() / template_ID() / template_any_ID() don't really
    # work for custom generic ID properties, so we have to use a workaround.
    # Adapted from https://blender.stackexchange.com/questions/292054/
    def custom_any_ID(layout, data, property, type_property, text=None, text_ctxt="", translate=True):
        if text is None: text = data.bl_rna.properties[property].name
        
        if text:
            split = layout.split(factor=0.33)
            row = split.row()
            row.label(text=text, text_ctxt=text_ctxt, translate=translate)
            row = split.row(align=True)
        else:
            row = layout.row(align=True)
        
        id_type = getattr(data, type_property)
        id_type_info = IDTypes.map("id_type").get(id_type)
        if id_type_info:
            collection_name = id_type_info.data_name
            if collection_name:
                icon = data.bl_rna.properties[type_property].enum_items[id_type].icon
                if icon == 'NONE': icon = id_type_info.icon
                
                # ID-Type Selector - just have a menu of icons
                # HACK: special group just for the enum,
                # otherwise we get ugly layout with text included too...
                sub = row.row(align=True)
                sub.alignment = 'LEFT'
                sub.prop(data, type_property, icon=icon, icon_only=True)
                
                # ID-Block Selector - just use pointer widget...
                # HACK: special group to counteract the effects of the previous enum,
                # which now pushes everything too far right.
                sub = row.row(align=True)
                sub.alignment = 'EXPAND'
                sub.prop_search(data, property, bpy.data, collection_name, text="")
            else:
                row.label(text=f"{id_type_info.type} not recognized", icon='ERROR')
        else:
            row.label(text=f"{id_type} not recognized", icon='ERROR')

#============================================================================#
