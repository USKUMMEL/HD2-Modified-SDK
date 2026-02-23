# ============================================================================== 
# Particle Modder - Blender Integration (minimal core)
# ============================================================================== 

#region Imports
import os
import struct
import bpy
from bpy.props import (
    StringProperty,
    IntProperty,
    FloatProperty,
    BoolProperty,
    PointerProperty,
)
from bpy.types import Operator, Panel, PropertyGroup
from bpy_extras.io_utils import ImportHelper, ExportHelper
#endregion

#region State
class _ParticleModderState:
    def __init__(self):
        self.filepath = ""
        self.data = None
        self.version = 0

STATE = _ParticleModderState()
#endregion

#region Helpers
_VALID_VERSIONS = {0x71, 0x6F, 0x6E, 0x6D}


def _parse_header(data: bytearray):
    if len(data) < 28:
        return None
    version = struct.unpack_from("<I", data, 0)[0]
    if version not in _VALID_VERSIONS:
        return None
    min_lifetime = struct.unpack_from("<f", data, 4)[0]
    max_lifetime = struct.unpack_from("<f", data, 8)[0]
    num_variables = struct.unpack_from("<I", data, 20)[0]
    num_systems = struct.unpack_from("<I", data, 24)[0]
    return {
        "version": version,
        "min_lifetime": min_lifetime,
        "max_lifetime": max_lifetime,
        "num_variables": num_variables,
        "num_systems": num_systems,
    }


def _apply_min_max(data: bytearray, min_lifetime: float, max_lifetime: float):
    struct.pack_into("<f", data, 4, float(min_lifetime))
    struct.pack_into("<f", data, 8, float(max_lifetime))
#endregion

#region Properties
class Hd2ParticleModderSettings(PropertyGroup):
    filepath: StringProperty(name="File", default="")
    version: IntProperty(name="Version", default=0)
    min_lifetime: FloatProperty(name="Min Lifetime", default=0.0)
    max_lifetime: FloatProperty(name="Max Lifetime", default=0.0)
    num_variables: IntProperty(name="Variables", default=0)
    num_systems: IntProperty(name="Particle Systems", default=0)
    has_data: BoolProperty(name="Has Data", default=False)
#endregion

#region Operators
class HD2_OT_ParticleModderLoad(Operator, ImportHelper):
    bl_idname = "hd2.particle_modder_load"
    bl_label = "Load Particle File"
    bl_options = {"REGISTER", "UNDO"}

    filter_glob: StringProperty(default="*.*", options={"HIDDEN"})

    def execute(self, context):
        settings = context.scene.Hd2ParticleModderSettings
        try:
            with open(self.filepath, "rb") as f:
                data = bytearray(f.read())
        except OSError as exc:
            self.report({"ERROR"}, f"Failed to read file: {exc}")
            return {"CANCELLED"}

        header = _parse_header(data)
        if header is None:
            self.report({"ERROR"}, "Unsupported or invalid particle file")
            return {"CANCELLED"}

        STATE.filepath = self.filepath
        STATE.data = data
        STATE.version = header["version"]

        settings.filepath = self.filepath
        settings.version = header["version"]
        settings.min_lifetime = header["min_lifetime"]
        settings.max_lifetime = header["max_lifetime"]
        settings.num_variables = header["num_variables"]
        settings.num_systems = header["num_systems"]
        settings.has_data = True

        return {"FINISHED"}


class HD2_OT_ParticleModderApply(Operator):
    bl_idname = "hd2.particle_modder_apply"
    bl_label = "Apply Changes"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.Hd2ParticleModderSettings
        if not settings.has_data or STATE.data is None:
            self.report({"ERROR"}, "No particle file loaded")
            return {"CANCELLED"}

        _apply_min_max(STATE.data, settings.min_lifetime, settings.max_lifetime)
        return {"FINISHED"}


class HD2_OT_ParticleModderSave(Operator, ExportHelper):
    bl_idname = "hd2.particle_modder_save"
    bl_label = "Save Particle File"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ""
    filter_glob: StringProperty(default="*.*", options={"HIDDEN"})

    def execute(self, context):
        settings = context.scene.Hd2ParticleModderSettings
        if not settings.has_data or STATE.data is None:
            self.report({"ERROR"}, "No particle file loaded")
            return {"CANCELLED"}

        _apply_min_max(STATE.data, settings.min_lifetime, settings.max_lifetime)

        try:
            with open(self.filepath, "wb") as f:
                f.write(STATE.data)
        except OSError as exc:
            self.report({"ERROR"}, f"Failed to write file: {exc}")
            return {"CANCELLED"}

        return {"FINISHED"}
#endregion

#region Panels
class HD2_PT_ParticleModder(Panel):
    bl_label = "HD2 Particle Modder"
    bl_idname = "HD2_PT_particle_modder"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "HD2"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.Hd2ParticleModderSettings

        layout.operator("hd2.particle_modder_load", icon="FILE_FOLDER")

        col = layout.column()
        col.enabled = settings.has_data

        col.prop(settings, "filepath")
        col.prop(settings, "version")
        col.prop(settings, "num_variables")
        col.prop(settings, "num_systems")
        col.prop(settings, "min_lifetime")
        col.prop(settings, "max_lifetime")

        row = col.row()
        row.operator("hd2.particle_modder_apply", icon="CHECKMARK")
        row.operator("hd2.particle_modder_save", icon="FILE_TICK")
#endregion

CLASSES = (
    Hd2ParticleModderSettings,
    HD2_OT_ParticleModderLoad,
    HD2_OT_ParticleModderApply,
    HD2_OT_ParticleModderSave,
    HD2_PT_ParticleModder,
)


def register_properties():
    bpy.types.Scene.Hd2ParticleModderSettings = PointerProperty(type=Hd2ParticleModderSettings)


def unregister_properties():
    if hasattr(bpy.types.Scene, "Hd2ParticleModderSettings"):
        del bpy.types.Scene.Hd2ParticleModderSettings

