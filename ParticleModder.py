# ============================================================================== 
# Particle Modder - Blender Integration (minimal core)
# ============================================================================== 

#region Imports
import os
import struct
import json
import gpu
from gpu_extras.batch import batch_for_shader
import blf
from mathutils import Matrix, Euler
from .utils.constants import ParticleID
import bpy
from bpy.props import (
    StringProperty,
    IntProperty,
    FloatProperty,
    BoolProperty,
    PointerProperty,
    CollectionProperty,
    EnumProperty,
    FloatVectorProperty,
)
from bpy.types import Operator, Panel, PropertyGroup, UIList, Menu
from bpy_extras.io_utils import ImportHelper, ExportHelper
#endregion

#region State
class _ParticleModderState:
    def __init__(self):
        self.filepath = ""
        self.data = None
        self.version = 0
        self.graph_curve = None
        self.graph_draw_handler = None
        self.graph_modal_running = False
        self.graph_drag_index = -1
        self.graph_rect = (40, 60, 360, 260)
        self.loaded_cache = {}
        self.suspend_selection_sync = False


def _cache_current(settings, flush=True):
    if STATE.data is None or not STATE.filepath:
        return
    if flush:
        try:
            _apply_settings_to_state_data_in_place(settings)
        except Exception:
            # Never break particle switching because a stale offset cannot be written.
            pass
    STATE.loaded_cache[STATE.filepath] = {
        "label": STATE.filepath,
        "data": bytearray(STATE.data),
        "file_id": settings.entry_file_id,
        "type_id": settings.entry_type_id,
        "is_archive": settings.is_archive,
        "selected_cells": settings.selected_cells,
        "last_selected_cell": settings.last_selected_cell,
    }


def _on_particle_index_change(scene, context):
    if STATE.suspend_selection_sync:
        return
    list_id = f"list_{ParticleID}"
    index_id = f"index_{ParticleID}"
    if not hasattr(scene, list_id) or not hasattr(scene, index_id):
        return
    mat_list = getattr(scene, list_id)
    mat_index = getattr(scene, index_id)
    if not mat_list or mat_index < 0 or mat_index >= len(mat_list):
        return
    entry = mat_list[mat_index]
    settings = getattr(scene, "Hd2ParticleModderSettings", None)
    if settings is not None:
        try:
            STATE.suspend_selection_sync = True
            settings.loaded_dump_particles_index = -1
        finally:
            STATE.suspend_selection_sync = False
    try:
        bpy.ops.helldiver2.particle_modder_edit(object_id=entry.item_name)
    except Exception:
        return


def _on_loaded_particle_index_change(settings, context):
    if STATE.suspend_selection_sync:
        return
    if settings.particle_source_tab != "DUMP":
        return
    idx = settings.loaded_particles_index
    if idx < 0 or idx >= len(settings.loaded_particles):
        return
    item = settings.loaded_particles[idx]
    if item.is_archive:
        return
    if item.key not in STATE.loaded_cache:
        return
    try:
        bpy.ops.hd2.particle_loaded_select(key=item.key)
    except Exception:
        return


def _on_loaded_dump_particle_index_change(settings, context):
    if STATE.suspend_selection_sync:
        return
    if settings.particle_source_tab != "DUMP":
        return
    idx = settings.loaded_dump_particles_index
    if idx < 0 or idx >= len(settings.loaded_dump_particles):
        return
    item = settings.loaded_dump_particles[idx]
    if item.key not in STATE.loaded_cache:
        return
    scene = context.scene
    archive_index_id = f"index_{ParticleID}"
    if hasattr(scene, archive_index_id):
        try:
            STATE.suspend_selection_sync = True
            setattr(scene, archive_index_id, -1)
        finally:
            STATE.suspend_selection_sync = False
    try:
        bpy.ops.hd2.particle_loaded_select(key=item.key)
    except Exception:
        return

STATE = _ParticleModderState()
#endregion

#region Helpers
_CURRENT_PARTICLE_EFFECT_VERSION = 0x72
_VALID_VERSIONS = {_CURRENT_PARTICLE_EFFECT_VERSION, 0x71, 0x6F, 0x6E, 0x6D}
_GRAPH_POINTS = 10
_GRAPH_BYTES = 4 * _GRAPH_POINTS * 2
_COLOR_GRAPH_BYTES = (4 * _GRAPH_POINTS) + (4 * _GRAPH_POINTS * 3)
_BURST_POINTS = 10

_EMITTER_BURST = 0x0C
_EMITTER_RATE = 0x0B

_VIS_BILLBOARD = 0
_VIS_LIGHT = 1
_VIS_MESH = 2
_VIS_UNKNOWN3 = 3
_VIS_UNKNOWN4 = 4

# Latest upstream modder no longer parses/writes emitter blocks with heuristics.
# Rewriting them here is unsafe and can corrupt particle binaries.
_ENABLE_EXPERIMENTAL_EMITTER_WRITES = False


class _PMStream:
    def __init__(self, data: bytearray):
        self.data = data
        self.pos = 0

    def seek(self, pos):
        self.pos = pos
        if self.pos > len(self.data):
            self.data += bytearray(self.pos - len(self.data))

    def tell(self):
        return self.pos

    def read(self, length):
        if self.pos + length > len(self.data):
            raise Exception("reading past end of stream")
        chunk = self.data[self.pos:self.pos + length]
        self.pos += length
        return bytes(chunk)

    def advance(self, length):
        self.pos += length
        if self.pos < 0:
            self.pos = 0
        if self.pos > len(self.data):
            self.data += bytearray(self.pos - len(self.data))

    def write(self, chunk):
        length = len(chunk)
        end = self.pos + length
        if end > len(self.data):
            self.data += bytearray(end - len(self.data))
        self.data[self.pos:end] = bytearray(chunk)
        self.pos = end

    def uint32_read(self):
        value = struct.unpack_from("<I", self.data, self.pos)[0]
        self.pos += 4
        return value

    def uint64_read(self):
        value = struct.unpack_from("<Q", self.data, self.pos)[0]
        self.pos += 8
        return value

    def float32_read(self):
        value = struct.unpack_from("<f", self.data, self.pos)[0]
        self.pos += 4
        return value


class Graph:
    def __init__(self):
        self.x = []
        self.y = []

    def from_memory_stream(self, stream: _PMStream):
        self.x = [stream.float32_read() for _ in range(_GRAPH_POINTS)]
        self.y = [stream.float32_read() for _ in range(_GRAPH_POINTS)]

    def write_to_memory_stream(self, stream: _PMStream):
        stream.write(struct.pack("<10f", *self.x))
        stream.write(struct.pack("<10f", *self.y))


class ColorGraph:
    def __init__(self):
        self.x = []
        self.y = []

    def from_memory_stream(self, stream: _PMStream):
        self.x = [stream.float32_read() for _ in range(_GRAPH_POINTS)]
        self.y = [[stream.float32_read(), stream.float32_read(), stream.float32_read()] for _ in range(_GRAPH_POINTS)]

    def write_to_memory_stream(self, stream: _PMStream):
        stream.write(struct.pack("<10f", *self.x))
        for color in self.y:
            stream.write(struct.pack("<fff", *color))


class Visualizer:
    BILLBOARD = 0
    LIGHT = 1
    MESH = 2
    UNKNOWN3 = 3
    UNKNOWN4 = 4

    def from_memory_stream(self, stream: _PMStream):
        self.visualizer_type = stream.uint32_read()
        self.unk1 = 0
        self.unk2 = 0
        self.material_id = 0
        self.unit_id = 0
        self.mesh_id = 0
        self.data = b""
        if self.visualizer_type == Visualizer.BILLBOARD:
            self.unk1 = stream.uint32_read()
            self.unk2 = stream.uint32_read()
            self.material_id = stream.uint64_read()
            self.data = stream.read(240)
        elif self.visualizer_type == Visualizer.LIGHT:
            self.data = stream.read(256)
        elif self.visualizer_type == Visualizer.MESH:
            self.unit_id = stream.uint64_read()
            self.mesh_id = stream.uint64_read()
            self.material_id = stream.uint64_read()
            self.data = stream.read(224)
        elif self.visualizer_type == Visualizer.UNKNOWN3:
            self.unk1 = stream.uint32_read()
            self.unk2 = stream.uint32_read()
            self.material_id = stream.uint64_read()
            self.data = stream.read(232)
        elif self.visualizer_type == Visualizer.UNKNOWN4:
            self.material_id = stream.uint64_read()
            self.data = stream.read(248)

    def write_to_memory_stream(self, stream: _PMStream):
        if self.visualizer_type == Visualizer.BILLBOARD:
            stream.write(struct.pack("<IIIQ", self.visualizer_type, self.unk1, self.unk2, self.material_id))
            stream.write(self.data)
        elif self.visualizer_type == Visualizer.LIGHT:
            stream.write(struct.pack("<I", self.visualizer_type))
            stream.write(self.data)
        elif self.visualizer_type == Visualizer.MESH:
            stream.write(struct.pack("<IQQQ", self.visualizer_type, self.unit_id, self.mesh_id, self.material_id))
            stream.write(self.data)
        elif self.visualizer_type == Visualizer.UNKNOWN3:
            stream.write(struct.pack("<IIIQ", self.visualizer_type, self.unk1, self.unk2, self.material_id))
            stream.write(self.data)
        elif self.visualizer_type == Visualizer.UNKNOWN4:
            stream.write(struct.pack("<IQ", self.visualizer_type, self.material_id))
            stream.write(self.data)


class _EmitterPositionModel:
    def __init__(self):
        self.position = [0.0, 0.0, 0.0]

    @classmethod
    def from_bytes(cls, data):
        model = cls()
        model.position = list(struct.unpack("<fff", data[0:12]))
        return model

    def to_bytes(self):
        return struct.pack("<fff", *self.position)


class _EmitterRotationModel:
    def __init__(self):
        self.rotation_euler = (0.0, 0.0, 0.0)

    @classmethod
    def from_bytes(cls, data):
        model = cls()
        m0 = struct.unpack("<fff", data[0:12])
        m1 = struct.unpack("<fff", data[16:28])
        m2 = struct.unpack("<fff", data[32:44])
        try:
            model.rotation_euler = tuple(Matrix((m0, m1, m2)).to_euler("XYZ"))
        except Exception:
            model.rotation_euler = (0.0, 0.0, 0.0)
        return model

    def to_bytes(self):
        matrix = Euler(self.rotation_euler, "XYZ").to_matrix()
        row0 = struct.pack("<fff", *matrix[0])
        row1 = struct.pack("<fff", *matrix[1])
        row2 = struct.pack("<fff", *matrix[2])
        padding = bytearray(4)
        return row0 + padding + row1 + padding + row2 + padding


class _ParticleEffectVariableModel:
    def __init__(self):
        self.name_hash = 0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0


class _ParticleSystemModel:
    def __init__(self, version):
        self.version = version
        self.offset = 0
        self.max_num_particles = 0
        self.num_components = 0
        self.non_rendering = 0
        self.unk1 = b""
        self.unk2 = b""
        self.unk3 = b""
        self.unk4 = b""
        self.unk5 = b""
        self.rotation = _EmitterRotationModel()
        self.position = _EmitterPositionModel()
        self.component_list_offset = 0
        self.emitter_offset = 0
        self.visualizer_offset = 0
        self.size = 0
        self.component_chunk = bytearray()
        self.emitter_chunk = bytearray()
        self.visualizer = None
        self.scale_graphs = []
        self.opacity_graphs = []
        self.color_graphs = []
        self.graph_block_offsets = []
        self.scale_graph_offsets = []
        self.opacity_graph_offsets = []
        self.color_graph_offsets = []
        self.other_graphs = []
        self.other_graph_offsets = []
        self.emitter_offsets = []
        self.emitters = []

    def is_rendering(self):
        return self.non_rendering == 0

    def from_memory_stream(self, stream):
        self.scale_graphs.clear()
        self.opacity_graphs.clear()
        self.color_graphs.clear()
        self.graph_block_offsets.clear()
        self.scale_graph_offsets.clear()
        self.opacity_graph_offsets.clear()
        self.color_graph_offsets.clear()
        self.other_graphs.clear()
        self.other_graph_offsets.clear()
        self.emitters.clear()
        self.emitter_offsets.clear()

        self.offset = stream.tell()
        self.max_num_particles = stream.uint32_read()
        self.num_components = stream.uint32_read()
        self.unk1 = stream.read(68)
        self.non_rendering = stream.uint32_read()
        self.unk2 = stream.read(40)
        self.rotation = _EmitterRotationModel.from_bytes(stream.read(48))
        self.position = _EmitterPositionModel.from_bytes(stream.read(12))
        self.unk3 = stream.read(52)
        self.component_list_offset = stream.uint32_read()
        self.unk4 = stream.read(4)
        self.emitter_offset = stream.uint32_read()
        self.unk5 = stream.read(8)
        self.visualizer_offset = stream.uint32_read()
        self.size = stream.uint32_read()

        stream.seek(self.offset + self.component_list_offset)
        self.component_chunk = bytearray(stream.read(self.emitter_offset - self.component_list_offset))
        stream.seek(self.offset + self.emitter_offset)
        self.emitter_chunk = bytearray(stream.read(self.visualizer_offset - self.emitter_offset))

        if not self.is_rendering():
            stream.seek(self.offset + self.size)
            return
        if self.visualizer_offset == self.size:
            stream.seek(self.offset + self.size)
            return

        stream.seek(self.offset + self.visualizer_offset)
        self.visualizer = Visualizer()
        self.visualizer.from_memory_stream(stream)

        while stream.tell() < self.offset + self.size:
            component_type = stream.uint32_read()
            if component_type in (0x05, 0x04, 0x0F):
                subtype = stream.uint32_read()
                if subtype < 0x20:
                    stream.advance(-4)
                    continue
                stream.advance(-8)
            elif component_type == 0x00:
                continue
            elif component_type == 0x11:
                if stream.tell() + 284 < self.offset + self.size:
                    stream.advance(284)
                continue
            elif component_type == 0x0B:
                stream.advance(24)
                continue
            else:
                continue

            if stream.tell() + 16 > self.offset + self.size:
                break

            component_type = [stream.uint32_read() for _ in range(4)]
            if component_type[0] == 0x04 and component_type[1] >= 0x20:
                stream.advance(4)
                self.other_graph_offsets.append(stream.tell() - self.offset)
                graph = Graph()
                graph.from_memory_stream(stream)
                graph.from_memory_stream(stream)
                self.other_graphs.append(graph)
                stream.advance(8)
            elif component_type[0] == 0x05 and component_type[1] >= 0x20:
                stream.advance(-4)
                self.graph_block_offsets.append(stream.tell() - self.offset)
                self.scale_graph_offsets.append(stream.tell() - self.offset)
                scale = Graph()
                scale.from_memory_stream(stream)
                scale.from_memory_stream(stream)
                self.scale_graphs.append(scale)
                self.opacity_graph_offsets.append(stream.tell() - self.offset)
                opacity = Graph()
                opacity.from_memory_stream(stream)
                opacity.from_memory_stream(stream)
                self.opacity_graphs.append(opacity)
                self.color_graph_offsets.append(stream.tell() - self.offset)
                color = ColorGraph()
                color.from_memory_stream(stream)
                self.color_graphs.append(color)
                stream.advance(16)
            elif component_type[1] == 0x05 and component_type[2] >= 0x20:
                self.graph_block_offsets.append(stream.tell() - self.offset)
                self.scale_graph_offsets.append(stream.tell() - self.offset)
                scale = Graph()
                scale.from_memory_stream(stream)
                scale.from_memory_stream(stream)
                self.scale_graphs.append(scale)
                self.opacity_graph_offsets.append(stream.tell() - self.offset)
                opacity = Graph()
                opacity.from_memory_stream(stream)
                opacity.from_memory_stream(stream)
                self.opacity_graphs.append(opacity)
                self.color_graph_offsets.append(stream.tell() - self.offset)
                color = ColorGraph()
                color.from_memory_stream(stream)
                self.color_graphs.append(color)
                stream.advance(16)
            elif component_type[0] == 0x0F and component_type[1] >= 0x20:
                stream.advance(-4)
                self.graph_block_offsets.append(stream.tell() - self.offset)
                self.scale_graphs.append(None)
                self.scale_graph_offsets.append(-1)
                self.opacity_graph_offsets.append(stream.tell() - self.offset)
                opacity = Graph()
                opacity.from_memory_stream(stream)
                opacity.from_memory_stream(stream)
                self.opacity_graphs.append(opacity)
                self.color_graph_offsets.append(stream.tell() - self.offset)
                color = ColorGraph()
                color.from_memory_stream(stream)
                self.color_graphs.append(color)
                stream.advance(16)
            elif component_type[0] == 0x0B:
                stream.advance(12)

        stream.seek(self.offset + self.size)

    def write_to_memory_stream(self, stream):
        stream.write(struct.pack("<II", self.max_num_particles, self.num_components))
        stream.write(self.unk1)
        stream.write(struct.pack("<I", self.non_rendering))
        stream.write(self.unk2)
        stream.write(self.rotation.to_bytes())
        stream.write(self.position.to_bytes())
        stream.write(self.unk3)
        stream.write(struct.pack("<I", self.component_list_offset))
        stream.write(self.unk4)
        stream.write(struct.pack("<I", self.emitter_offset))
        stream.write(self.unk5)
        stream.write(struct.pack("<II", self.visualizer_offset, self.size))

        stream.seek(self.offset + self.component_list_offset)
        stream.write(self.component_chunk)
        stream.seek(self.offset + self.emitter_offset)
        stream.write(self.emitter_chunk)

        if self.non_rendering != 0:
            stream.seek(self.offset + self.size)
            return
        if self.visualizer_offset == self.size:
            stream.seek(self.offset + self.size)
            return

        for index, offset in enumerate(self.emitter_offsets):
            stream.seek(offset)
            self.emitters[index].write_to_memory_stream(stream)

        stream.seek(self.offset + self.visualizer_offset)
        if self.visualizer is not None:
            self.visualizer.write_to_memory_stream(stream)

        for index, offset in enumerate(self.graph_block_offsets):
            stream.seek(self.offset + offset)
            if self.scale_graphs[index] is not None:
                self.scale_graphs[index].write_to_memory_stream(stream)
                self.scale_graphs[index].write_to_memory_stream(stream)
            self.opacity_graphs[index].write_to_memory_stream(stream)
            self.opacity_graphs[index].write_to_memory_stream(stream)
            self.color_graphs[index].write_to_memory_stream(stream)

        for index, offset in enumerate(self.other_graph_offsets):
            stream.seek(self.offset + offset)
            self.other_graphs[index].write_to_memory_stream(stream)
            self.other_graphs[index].write_to_memory_stream(stream)


class _ParticleEffectModel:
    def __init__(self):
        self.variables = []
        self.particle_systems = []
        self.min_lifetime = 0.0
        self.max_lifetime = 0.0
        self.num_variables = 0
        self.num_particle_systems = 0
        self.version = 0

    def from_memory_stream(self, stream):
        self.variables.clear()
        self.particle_systems.clear()
        self.version = stream.uint32_read()
        if self.version not in _VALID_VERSIONS:
            raise Exception(f"Unsupported particle version: {hex(self.version)}")
        self.min_lifetime = stream.float32_read()
        self.max_lifetime = stream.float32_read()
        stream.advance(8)
        self.num_variables = stream.uint32_read()
        self.num_particle_systems = stream.uint32_read()
        stream.advance(44)
        if self.version in (0x6F, 0x71, 0x72):
            stream.advance(8)
        for _ in range(self.num_variables):
            var = _ParticleEffectVariableModel()
            var.name_hash = stream.uint32_read()
            self.variables.append(var)
        for variable in self.variables:
            variable.x = stream.float32_read()
            variable.y = stream.float32_read()
            variable.z = stream.float32_read()
        for _ in range(self.num_particle_systems):
            system = _ParticleSystemModel(self.version)
            system.from_memory_stream(stream)
            self.particle_systems.append(system)

    def write_to_memory_stream(self, stream):
        stream.seek(0)
        stream.write(struct.pack("<I", _CURRENT_PARTICLE_EFFECT_VERSION))
        stream.write(struct.pack("<ff", self.min_lifetime, self.max_lifetime))
        stream.advance(8)
        stream.write(int(self.num_variables).to_bytes(4, byteorder="little"))
        stream.write(int(self.num_particle_systems).to_bytes(4, byteorder="little"))
        if self.version in (0x6F, 0x71, 0x72):
            stream.advance(52)
        else:
            stream.advance(44)
            stream.data[stream.tell():stream.tell()] = bytearray(8)
            stream.advance(8)
            for system in self.particle_systems:
                system.offset += 8
        for variable in self.variables:
            stream.write(struct.pack("<I", int(variable.name_hash)))
        for variable in self.variables:
            stream.write(struct.pack("<fff", float(variable.x), float(variable.y), float(variable.z)))
        for system in self.particle_systems:
            stream.seek(system.offset)
            system.write_to_memory_stream(stream)
        if self.version < 0x71 and self.particle_systems:
            updated_offset = 0
            for system in self.particle_systems:
                if not system.is_rendering():
                    continue
                offset = system.offset + updated_offset + system.emitter_offset - 16
                stream.seek(offset)
                value = stream.uint32_read()
                stream.seek(stream.tell() + 12)
                if value == 8:
                    stream.seek(system.offset + updated_offset + 0xFC)
                    stream.write(struct.pack("<II", system.visualizer_offset + 16, system.size + 16))
                    stream.seek(system.offset + updated_offset + system.emitter_offset + 8)
                    stream.data[stream.tell():stream.tell()] = b"\xFF\xFF\xFF\xFF"
                    replace_offset = stream.data.find(
                        b"\x08\x00\x00\x00\x00\x00\x00\x00",
                        system.offset + system.emitter_offset + updated_offset,
                        system.offset + system.visualizer_offset + updated_offset,
                    )
                    if replace_offset == -1:
                        continue
                    stream.seek(replace_offset + 8)
                    replace_value = stream.uint32_read()
                    stream.data[replace_offset + 8:replace_offset + 8] = struct.pack(
                        "<III", replace_value + 0x10, replace_value + 0x14, replace_value + 0x18
                    )
                    updated_offset += 16
                else:
                    stream.seek(system.offset + updated_offset + 0xFC)
                    stream.write(struct.pack("<II", system.visualizer_offset + 4, system.size + 4))
                    stream.seek(system.offset + updated_offset + system.emitter_offset + 8)
                    stream.data[stream.tell():stream.tell()] = b"\xFF\xFF\xFF\xFF"
                    updated_offset += 4
        if self.version != _CURRENT_PARTICLE_EFFECT_VERSION:
            stream.seek(0)
            self.from_memory_stream(stream)
            self.version = _CURRENT_PARTICLE_EFFECT_VERSION


def _parse_particle_effect_model(data: bytearray):
    stream = _PMStream(bytearray(data))
    effect = _ParticleEffectModel()
    effect.from_memory_stream(stream)
    return effect


def _copy_graph_points_to_settings(item, graph):
    for p in range(_GRAPH_POINTS):
        pt = item.points.add()
        pt.x = float(graph.x[p])
        pt.y = float(graph.y[p])


def _copy_color_points_to_settings(item, graph):
    for p in range(_GRAPH_POINTS):
        pt = item.points.add()
        pt.x = float(graph.x[p])
        pt.r = float(graph.y[p][0])
        pt.g = float(graph.y[p][1])
        pt.b = float(graph.y[p][2])
        pt.color = (
            max(0.0, min(1.0, pt.r / 255.0)),
            max(0.0, min(1.0, pt.g / 255.0)),
            max(0.0, min(1.0, pt.b / 255.0)),
        )


def _apply_settings_graph_to_model(item, graph):
    graph.x = [float(item.points[i].x) if i < len(item.points) else 0.0 for i in range(_GRAPH_POINTS)]
    graph.y = [float(item.points[i].y) if i < len(item.points) else 0.0 for i in range(_GRAPH_POINTS)]


def _apply_settings_color_to_model(item, graph):
    graph.x = [float(item.points[i].x) if i < len(item.points) else 0.0 for i in range(_GRAPH_POINTS)]
    graph.y = []
    for i in range(_GRAPH_POINTS):
        if i < len(item.points):
            graph.y.append([
                float(item.points[i].r),
                float(item.points[i].g),
                float(item.points[i].b),
            ])
        else:
            graph.y.append([0.0, 0.0, 0.0])


def _pack_graph_from_settings(item):
    xs = [float(item.points[i].x) if i < len(item.points) else 0.0 for i in range(_GRAPH_POINTS)]
    ys = [float(item.points[i].y) if i < len(item.points) else 0.0 for i in range(_GRAPH_POINTS)]
    return struct.pack("<10f", *xs) + struct.pack("<10f", *ys)


def _pack_color_graph_from_settings(item):
    xs = [float(item.points[i].x) if i < len(item.points) else 0.0 for i in range(_GRAPH_POINTS)]
    payload = bytearray(struct.pack("<10f", *xs))
    for i in range(_GRAPH_POINTS):
        if i < len(item.points):
            payload += struct.pack(
                "<fff",
                float(item.points[i].r),
                float(item.points[i].g),
                float(item.points[i].b),
            )
        else:
            payload += struct.pack("<fff", 0.0, 0.0, 0.0)
    return bytes(payload)


def _write_bytes_at(data: bytearray, offset: int, payload: bytes):
    if offset < 0:
        return False
    end = offset + len(payload)
    if end > len(data):
        return False
    data[offset:end] = payload
    return True


def _write_transform_to_bytes(data: bytearray, transform):
    matrix = Euler(
        (
            float(transform.rot_x),
            float(transform.rot_y),
            float(transform.rot_z),
        ),
        "XYZ",
    ).to_matrix()
    rotation_bytes = (
        struct.pack("<fff", *matrix[0]) + bytearray(4) +
        struct.pack("<fff", *matrix[1]) + bytearray(4) +
        struct.pack("<fff", *matrix[2]) + bytearray(4)
    )
    position_bytes = struct.pack(
        "<fff",
        float(transform.pos_x),
        float(transform.pos_y),
        float(transform.pos_z),
    )
    _write_bytes_at(data, int(transform.rotation_offset), rotation_bytes)
    _write_bytes_at(data, int(transform.position_offset), position_bytes)


class _ParticleSystemGraphInfo:
    def __init__(self):
        self.offset = 0
        self.size = 0
        self.non_rendering = 0
        self.rotation_offset = 0
        self.position_offset = 0
        self.emitter_offset = 0
        self.visualizer_offset = 0
        self.scale_graphs = []
        self.scale_offsets = []
        self.opacity_graphs = []
        self.opacity_offsets = []
        self.color_graphs = []
        self.color_offsets = []
        self.other_graphs = []
        self.other_offsets = []

    def from_memory_stream(self, stream: _PMStream):
        self.scale_graphs.clear()
        self.opacity_graphs.clear()
        self.color_graphs.clear()
        self.other_graphs.clear()
        self.scale_offsets.clear()
        self.opacity_offsets.clear()
        self.color_offsets.clear()
        self.other_offsets.clear()

        self.offset = stream.tell()
        data_len = len(stream.data)
        # Minimal particle-system header size used by parser.
        if self.offset + 0x104 > data_len:
            raise Exception("particle system header truncated")
        stream.uint32_read()  # max_num_particles
        stream.uint32_read()  # num_components
        stream.read(68)
        self.non_rendering = stream.uint32_read()
        stream.read(40)
        self.rotation_offset = stream.tell()
        stream.read(48)  # rotation
        self.position_offset = stream.tell()
        stream.read(12)  # position
        stream.read(52)
        component_list_offset = stream.uint32_read()
        stream.read(4)
        self.emitter_offset = stream.uint32_read()
        stream.read(8)
        self.visualizer_offset = stream.uint32_read()
        self.size = stream.uint32_read()

        # Guard against corrupted or shifted offsets.
        if self.size <= 0:
            raise Exception("invalid particle system size")
        if self.offset + self.size > data_len:
            self.size = data_len - self.offset
        if component_list_offset < 0 or self.emitter_offset < 0 or self.visualizer_offset < 0:
            raise Exception("negative offsets in particle system")
        if not (component_list_offset <= self.emitter_offset <= self.visualizer_offset <= self.size):
            # Skip this system safely if header layout doesn't match current parser.
            stream.seek(self.offset + self.size)
            self.non_rendering = 1
            return

        stream.seek(self.offset + component_list_offset)
        stream.read(self.emitter_offset - component_list_offset)
        stream.seek(self.offset + self.emitter_offset)
        stream.read(self.visualizer_offset - self.emitter_offset)

        if self.non_rendering != 0 or self.visualizer_offset == self.size:
            stream.seek(self.offset + self.size)
            return

        stream.seek(self.offset + self.visualizer_offset)
        Visualizer().from_memory_stream(stream)

        while stream.tell() < self.offset + self.size:
            component_type = stream.uint32_read()
            if component_type in (0x05, 0x04, 0x0F):
                subtype = stream.uint32_read()
                if subtype < 0x20:
                    stream.advance(-4)
                    continue
                stream.advance(-8)
            elif component_type == 0x00:
                continue
            elif component_type == 0x11:
                if stream.tell() + 284 < self.offset + self.size:
                    stream.advance(284)
                continue
            elif component_type == 0x0B:
                stream.advance(24)
                continue
            else:
                continue

            if stream.tell() + 16 > self.offset + self.size:
                break

            component_type = [stream.uint32_read() for _ in range(4)]
            if component_type[0] == 0x04 and component_type[1] >= 0x20:
                stream.advance(4)
                self.other_offsets.append(stream.tell() - self.offset)
                g = Graph()
                g.from_memory_stream(stream)
                g.from_memory_stream(stream)
                self.other_graphs.append(g)
                stream.advance(8)
            elif component_type[0] == 0x05 and component_type[1] >= 0x20:
                stream.advance(-4)
                self.scale_offsets.append(stream.tell() - self.offset)
                scale = Graph()
                scale.from_memory_stream(stream)
                scale.from_memory_stream(stream)
                self.scale_graphs.append(scale)

                self.opacity_offsets.append(stream.tell() - self.offset)
                opacity = Graph()
                opacity.from_memory_stream(stream)
                opacity.from_memory_stream(stream)
                self.opacity_graphs.append(opacity)

                self.color_offsets.append(stream.tell() - self.offset)
                color = ColorGraph()
                color.from_memory_stream(stream)
                self.color_graphs.append(color)
                stream.advance(16)
            elif component_type[1] == 0x05 and component_type[2] >= 0x20:
                self.scale_offsets.append(stream.tell() - self.offset)
                scale = Graph()
                scale.from_memory_stream(stream)
                scale.from_memory_stream(stream)
                self.scale_graphs.append(scale)

                self.opacity_offsets.append(stream.tell() - self.offset)
                opacity = Graph()
                opacity.from_memory_stream(stream)
                opacity.from_memory_stream(stream)
                self.opacity_graphs.append(opacity)

                self.color_offsets.append(stream.tell() - self.offset)
                color = ColorGraph()
                color.from_memory_stream(stream)
                self.color_graphs.append(color)
                stream.advance(16)
            elif component_type[0] == 0x0F and component_type[1] >= 0x20:
                stream.advance(-4)
                self.scale_graphs.append(None)
                self.scale_offsets.append(-1)

                self.opacity_offsets.append(stream.tell() - self.offset)
                opacity = Graph()
                opacity.from_memory_stream(stream)
                opacity.from_memory_stream(stream)
                self.opacity_graphs.append(opacity)

                self.color_offsets.append(stream.tell() - self.offset)
                color = ColorGraph()
                color.from_memory_stream(stream)
                self.color_graphs.append(color)
                stream.advance(16)
            elif component_type[0] == 0x0B:
                stream.advance(12)

        stream.seek(self.offset + self.size)


def _variables_offset(version):
    offset = 0
    offset += 4  # version
    offset += 4  # min_lifetime
    offset += 4  # max_lifetime
    offset += 8  # unknown padding
    offset += 4  # num_variables
    offset += 4  # num_particle_systems
    offset += 44
    if version in (0x6F, 0x71, 0x72):
        offset += 8
    return offset


def _read_variables(data: bytearray, version: int, num_variables: int):
    offset = _variables_offset(version)
    hashes = []
    for i in range(num_variables):
        hashes.append(struct.unpack_from("<I", data, offset + i * 4)[0])
    vectors_offset = offset + (num_variables * 4)
    vectors = []
    for i in range(num_variables):
        x, y, z = struct.unpack_from("<fff", data, vectors_offset + i * 12)
        vectors.append((x, y, z))
    return hashes, vectors


def _scan_graphs(data: bytearray, version: int, num_variables: int, num_systems: int):
    stream = _PMStream(data)
    start = _variables_offset(version) + (num_variables * 4) + (num_variables * 12)
    stream.seek(start)
    infos = []
    for _ in range(num_systems):
        info = _ParticleSystemGraphInfo()
        try:
            info.from_memory_stream(stream)
        except Exception:
            break
        infos.append(info)
    return infos


class _SystemTransformInfo:
    def __init__(self):
        self.system_index = 0
        self.rotation_offset = 0
        self.position_offset = 0
        self.rotation_euler = (0.0, 0.0, 0.0)
        self.position = (0.0, 0.0, 0.0)


def _scan_transforms(data: bytearray, graph_infos):
    transforms = []
    data_len = len(data)
    for system_index, info in enumerate(graph_infos):
        t = _SystemTransformInfo()
        t.system_index = system_index
        # Use the exact offsets captured by particle-system parser.
        t.rotation_offset = int(info.rotation_offset) if info.rotation_offset > 0 else (info.offset + 0x78)
        t.position_offset = int(info.position_offset) if info.position_offset > 0 else (info.offset + 0xA8)
        if t.rotation_offset + 44 > data_len or t.position_offset + 12 > data_len:
            continue
        try:
            m0 = struct.unpack_from("<fff", data, t.rotation_offset + 0)
            m1 = struct.unpack_from("<fff", data, t.rotation_offset + 16)
            m2 = struct.unpack_from("<fff", data, t.rotation_offset + 32)
            t.rotation_euler = tuple(Matrix((m0, m1, m2)).to_euler("XYZ"))
        except Exception:
            t.rotation_euler = (0.0, 0.0, 0.0)
        try:
            t.position = struct.unpack_from("<fff", data, t.position_offset)
        except Exception:
            t.position = (0.0, 0.0, 0.0)
        transforms.append(t)

    return transforms


class _EmitterInfo:
    def __init__(self):
        self.system_index = 0
        self.offset = 0
        self.emitter_type = 0
        self.initial_rate_min = 0.0
        self.initial_rate_max = 0.0
        self.rate_x = []
        self.rate_y = []
        self.burst_times = []
        self.burst_num_a = []
        self.burst_num_b = []


def _scan_emitters(data: bytearray, graph_infos):
    emitters = []
    data_len = len(data)
    for system_index, info in enumerate(graph_infos):
        if info.non_rendering != 0:
            continue
        start = info.offset + info.emitter_offset
        end = info.offset + info.visualizer_offset
        if start < 0 or end < 0 or start >= data_len:
            continue
        end = min(end, data_len)
        if start >= end:
            continue
        pos = start
        while pos + 4 <= end:
            emitter_type = struct.unpack_from("<I", data, pos)[0]
            if emitter_type == _EMITTER_BURST and pos + 4 + (_BURST_POINTS * 12) <= end:
                e = _EmitterInfo()
                e.system_index = system_index
                e.offset = pos
                e.emitter_type = emitter_type
                cursor = pos + 4
                for _ in range(_BURST_POINTS):
                    t = struct.unpack_from("<f", data, cursor)[0]
                    a, b = struct.unpack_from("<II", data, cursor + 4)
                    e.burst_times.append(t)
                    e.burst_num_a.append(a)
                    e.burst_num_b.append(b)
                    cursor += 12
                emitters.append(e)
                pos = cursor
                continue
            if emitter_type == _EMITTER_RATE and pos + 4 + 8 + _GRAPH_BYTES <= end:
                e = _EmitterInfo()
                e.system_index = system_index
                e.offset = pos
                e.emitter_type = emitter_type
                e.initial_rate_min = struct.unpack_from("<f", data, pos + 4)[0]
                e.initial_rate_max = struct.unpack_from("<f", data, pos + 8)[0]
                graph_base = pos + 12
                e.rate_x = [struct.unpack_from("<f", data, graph_base + (i * 4))[0] for i in range(_GRAPH_POINTS)]
                y_base = graph_base + (_GRAPH_POINTS * 4)
                e.rate_y = [struct.unpack_from("<f", data, y_base + (i * 4))[0] for i in range(_GRAPH_POINTS)]
                emitters.append(e)
                pos = graph_base + _GRAPH_BYTES
                continue
            pos += 4
    return emitters


class _VisualizerInfo:
    def __init__(self):
        self.system_index = 0
        self.offset = 0
        self.vis_type = 0
        self.unk1 = 0
        self.unk2 = 0
        self.material_id = 0
        self.unit_id = 0
        self.mesh_id = 0


def _scan_visualizers(data: bytearray, graph_infos):
    vis_list = []
    for system_index, info in enumerate(graph_infos):
        if info.non_rendering != 0 or info.visualizer_offset == 0:
            continue
        offset = info.offset + info.visualizer_offset
        if offset + 4 > len(data):
            continue
        vis_type = struct.unpack_from("<I", data, offset)[0]
        v = _VisualizerInfo()
        v.system_index = system_index
        v.offset = offset
        v.vis_type = vis_type
        if vis_type == _VIS_BILLBOARD:
            v.unk1, v.unk2 = struct.unpack_from("<II", data, offset + 4)
            v.material_id = struct.unpack_from("<Q", data, offset + 12)[0]
        elif vis_type == _VIS_LIGHT:
            pass
        elif vis_type == _VIS_MESH:
            v.unit_id = struct.unpack_from("<Q", data, offset + 4)[0]
            v.mesh_id = struct.unpack_from("<Q", data, offset + 12)[0]
            v.material_id = struct.unpack_from("<Q", data, offset + 20)[0]
        elif vis_type == _VIS_UNKNOWN3:
            v.unk1, v.unk2 = struct.unpack_from("<II", data, offset + 4)
            v.material_id = struct.unpack_from("<Q", data, offset + 12)[0]
        elif vis_type == _VIS_UNKNOWN4:
            v.material_id = struct.unpack_from("<Q", data, offset + 4)[0]
        vis_list.append(v)
    return vis_list


def _parse_selected_cells_value(selected_cells):
    if not selected_cells:
        return []
    return [s for s in selected_cells.split("|") if s]


def _selected_color_cells_map_from_cells(cells):
    mapping = {}
    for key in cells:
        parts = key.split(":")
        if len(parts) != 4:
            continue
        group, gidx, pidx, field = parts
        if group != "color" or field != "color":
            continue
        gidx_i = int(gidx)
        pidx_i = int(pidx)
        mapping.setdefault(gidx_i, set()).add(pidx_i)
    return mapping


def _apply_color_to_bytes(data: bytearray, version: int, num_variables: int, num_systems: int, selection_map, color_rgb):
    graph_infos = _scan_graphs(data, version, num_variables, num_systems)
    data_len = len(data)
    flat_offsets = []
    for info in graph_infos:
        for off in info.color_offsets:
            flat_offsets.append(info.offset + off)
    r, g, b = color_rgb
    for gidx, points in selection_map.items():
        if gidx < 0 or gidx >= len(flat_offsets):
            continue
        base = flat_offsets[gidx]
        color_base = base + (_GRAPH_POINTS * 4)
        for pidx in points:
            if pidx < 0 or pidx >= _GRAPH_POINTS:
                continue
            pos = color_base + (pidx * 12)
            if pos < 0 or pos + 12 > data_len:
                continue
            struct.pack_into("<fff", data, pos, float(r), float(g), float(b))


def _write_variables(data: bytearray, version: int, variables):
    num_variables = len(variables)
    offset = _variables_offset(version)
    data_len = len(data)
    for i, var in enumerate(variables):
        try:
            hash_value = int(var.name_hash, 0)
        except (ValueError, TypeError):
            hash_value = 0
        pos = offset + i * 4
        if pos < 0 or pos + 4 > data_len:
            break
        struct.pack_into("<I", data, offset + i * 4, hash_value)
    vectors_offset = offset + (num_variables * 4)
    for i, var in enumerate(variables):
        pos = vectors_offset + i * 12
        if pos < 0 or pos + 12 > data_len:
            break
        struct.pack_into("<fff", data, vectors_offset + i * 12, float(var.x), float(var.y), float(var.z))


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


def _legacy_version_fixup(data: bytearray, scan_version: int, num_variables: int, num_systems: int, source_version: int | None = None):
    # Keep behavior aligned with original tool for pre-0x71 layouts.
    if source_version is None:
        source_version = scan_version
    if source_version >= 0x71:
        return
    try:
        infos = _scan_graphs(data, scan_version, num_variables, num_systems)
    except Exception:
        return

    updated_offset = 0
    for info in infos:
        if info.non_rendering != 0:
            continue
        base = info.offset + updated_offset
        probe = base + info.emitter_offset - 16
        if probe < 0 or probe + 4 > len(data):
            continue
        value = struct.unpack_from("<I", data, probe)[0]

        header_fix = base + 0xFC
        if header_fix + 8 > len(data):
            continue

        if value == 8:
            struct.pack_into("<II", data, header_fix, info.visualizer_offset + 16, info.size + 16)
            ins_pos = base + info.emitter_offset + 8
            if 0 <= ins_pos <= len(data):
                data[ins_pos:ins_pos] = b"\xFF\xFF\xFF\xFF"
            find_start = base + info.emitter_offset
            find_end = base + info.visualizer_offset
            if 0 <= find_start < len(data) and find_start < find_end:
                marker = b"\x08\x00\x00\x00\x00\x00\x00\x00"
                replace_offset = data.find(marker, find_start, min(find_end, len(data)))
                if replace_offset != -1 and replace_offset + 12 <= len(data):
                    replace_value = struct.unpack_from("<I", data, replace_offset + 8)[0]
                    data[replace_offset + 8:replace_offset + 8] = struct.pack(
                        "<III", replace_value + 0x10, replace_value + 0x14, replace_value + 0x18
                    )
                    updated_offset += 16
        else:
            struct.pack_into("<II", data, header_fix, info.visualizer_offset + 4, info.size + 4)
            ins_pos = base + info.emitter_offset + 8
            if 0 <= ins_pos <= len(data):
                data[ins_pos:ins_pos] = b"\xFF\xFF\xFF\xFF"
                updated_offset += 4


def _upgrade_particle_effect_to_current_version(data: bytearray, source_version: int, num_variables: int, num_systems: int):
    changed = False
    if source_version not in _VALID_VERSIONS:
        return changed, source_version
    if source_version < 0x6F:
        insert_at = _variables_offset(source_version)
        if 0 <= insert_at <= len(data):
            data[insert_at:insert_at] = bytearray(8)
            changed = True
    if source_version != _CURRENT_PARTICLE_EFFECT_VERSION and len(data) >= 4:
        struct.pack_into("<I", data, 0, _CURRENT_PARTICLE_EFFECT_VERSION)
        changed = True
    if source_version < 0x71:
        before_len = len(data)
        _legacy_version_fixup(
            data,
            _CURRENT_PARTICLE_EFFECT_VERSION,
            num_variables,
            num_systems,
            source_version=source_version,
        )
        if len(data) != before_len:
            changed = True
    return changed, _CURRENT_PARTICLE_EFFECT_VERSION if changed else source_version


def _dump_particle_id_from_label(label: str):
    try:
        text = label if isinstance(label, str) else str(label)
    except Exception:
        text = ""
    # Handle archive-style labels (archive:12345) without path ops.
    if ":" in text and ("\\" not in text and "/" not in text):
        return text.split(":")[-1] or text
    base = os.path.basename(text)
    stem, _ext = os.path.splitext(base)
    return stem if stem else base
#endregion

class Hd2ParticleVariableItem(PropertyGroup):
    name_hash: StringProperty(name="Name Hash", default="")
    x: FloatProperty(name="X", default=0.0)
    y: FloatProperty(name="Y", default=0.0)
    z: FloatProperty(name="Z", default=0.0)


class HD2_UL_ParticleVariables(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "name_hash", text="")
        row.label(text=f"({item.x:.3f}, {item.y:.3f}, {item.z:.3f})")
#endregion

#region Color Sync
_COLOR_SYNC_GUARD = False


def _sync_color_to_rgb(self, context):
    global _COLOR_SYNC_GUARD
    if _COLOR_SYNC_GUARD:
        return
    _COLOR_SYNC_GUARD = True
    try:
        c0 = max(0.0, min(1.0, float(self.color[0])))
        c1 = max(0.0, min(1.0, float(self.color[1])))
        c2 = max(0.0, min(1.0, float(self.color[2])))
        if self.r != c0 or self.g != c1 or self.b != c2:
            self.r = c0 * 255.0
            self.g = c1 * 255.0
            self.b = c2 * 255.0
    finally:
        _COLOR_SYNC_GUARD = False


def _sync_rgb_to_color(self, context):
    global _COLOR_SYNC_GUARD
    if _COLOR_SYNC_GUARD:
        return
    _COLOR_SYNC_GUARD = True
    try:
        if self.color[0] != self.r or self.color[1] != self.g or self.color[2] != self.b:
            self.color = (
                max(0.0, min(1.0, float(self.r) / 255.0)),
                max(0.0, min(1.0, float(self.g) / 255.0)),
                max(0.0, min(1.0, float(self.b) / 255.0)),
            )
    finally:
        _COLOR_SYNC_GUARD = False
#endregion

#region Curve Mapping
def _load_curve_from_graph(settings, graph):
    curve = STATE.graph_curve
    if curve is None:
        # CurveMapping cannot be constructed directly in Blender 4.x.
        return False
    curve.initialize()
    points = curve.curves[0].points
    while len(points) > 0:
        points.remove(points[0])
    for p in graph.points:
        points.new(p.x, p.y)
    curve.update()
    return True


def _sync_graph_from_curve(settings, graph):
    curve = STATE.graph_curve
    if curve is None:
        return
    pts = curve.curves[0].points
    for i, p in enumerate(graph.points):
        if i >= len(pts):
            break
        p.x = pts[i].location[0]
        p.y = pts[i].location[1]


def _on_graph_index_change(self, context):
    if not self.graphs:
        return
    idx = self.graphs_index
    if idx < 0 or idx >= len(self.graphs):
        return
    _load_curve_from_graph(self, self.graphs[idx])
#endregion

#region Graph Canvas
def _graph_points_to_screen(points, rect):
    x0, y0, x1, y1 = rect
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    coords = []
    for p in points:
        px = max(0.0, min(1.0, p.x))
        py = max(0.0, min(1.0, p.y))
        x = x0 + (px * w)
        y = y0 + (py * h)
        coords.append((x, y))
    return coords


def _graph_screen_to_point(x, y, rect):
    x0, y0, x1, y1 = rect
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    nx = (x - x0) / w
    ny = (y - y0) / h
    return max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny))


def _draw_graph_callback():
    settings = bpy.context.scene.Hd2ParticleModderSettings
    if not settings.graphs:
        return
    idx = settings.graphs_index
    if idx < 0 or idx >= len(settings.graphs):
        return
    graph = settings.graphs[idx]
    rect = STATE.graph_rect

    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    # Draw background
    bg_coords = [(rect[0], rect[1]), (rect[2], rect[1]), (rect[2], rect[3]), (rect[0], rect[3])]
    batch = batch_for_shader(shader, "TRI_FAN", {"pos": bg_coords})
    shader.bind()
    shader.uniform_float("color", (0.08, 0.08, 0.08, 0.6))
    batch.draw(shader)

    # Draw border
    border = [(rect[0], rect[1]), (rect[2], rect[1]), (rect[2], rect[3]), (rect[0], rect[3]), (rect[0], rect[1])]
    batch = batch_for_shader(shader, "LINE_STRIP", {"pos": border})
    shader.uniform_float("color", (0.4, 0.4, 0.4, 1.0))
    batch.draw(shader)

    # Grid lines
    grid = []
    for t in (0.25, 0.5, 0.75):
        x = rect[0] + (rect[2] - rect[0]) * t
        grid.append([(x, rect[1]), (x, rect[3])])
        y = rect[1] + (rect[3] - rect[1]) * t
        grid.append([(rect[0], y), (rect[2], y)])
    shader.uniform_float("color", (0.2, 0.2, 0.2, 0.8))
    for line in grid:
        batch = batch_for_shader(shader, "LINE_STRIP", {"pos": line})
        batch.draw(shader)

    # Axis labels
    blf.position(0, rect[0], rect[1] - 14, 0)
    blf.size(0, 12)
    blf.draw(0, "0.0")
    blf.position(0, rect[2] - 18, rect[1] - 14, 0)
    blf.draw(0, "1.0")
    blf.position(0, rect[0] - 20, rect[1] - 2, 0)
    blf.draw(0, "0.0")
    blf.position(0, rect[0] - 20, rect[3] - 6, 0)
    blf.draw(0, "1.0")

    coords = _graph_points_to_screen(graph.points, rect)
    if len(coords) >= 2:
        batch = batch_for_shader(shader, "LINE_STRIP", {"pos": coords})
        shader.uniform_float("color", (0.2, 0.7, 1.0, 1.0))
        batch.draw(shader)

    # Draw points
    for i, (x, y) in enumerate(coords):
        size = 4 if i != STATE.graph_drag_index else 6
        point_rect = [(x - size, y - size), (x + size, y - size), (x + size, y + size), (x - size, y + size)]
        batch = batch_for_shader(shader, "TRI_FAN", {"pos": point_rect})
        shader.uniform_float("color", (1.0, 0.6, 0.1, 1.0))
        batch.draw(shader)
#endregion

#region Graphs
class Hd2GraphPoint(PropertyGroup):
    x: FloatProperty(name="X", default=0.0)
    y: FloatProperty(name="Y", default=0.0)


class Hd2GraphItem(PropertyGroup):
    label: StringProperty(name="Label", default="")
    kind: StringProperty(name="Kind", default="")
    offset: IntProperty(name="Offset", default=0)
    repeat_count: IntProperty(name="Repeat", default=1)
    points: CollectionProperty(type=Hd2GraphPoint)


class Hd2ColorPoint(PropertyGroup):
    x: FloatProperty(name="X", default=0.0)
    r: FloatProperty(name="R", default=0.0, update=_sync_rgb_to_color)
    g: FloatProperty(name="G", default=0.0, update=_sync_rgb_to_color)
    b: FloatProperty(name="B", default=0.0, update=_sync_rgb_to_color)
    color: FloatVectorProperty(
        name="Color",
        size=3,
        subtype="COLOR",
        default=(0.0, 0.0, 0.0),
        min=0.0,
        max=1.0,
        soft_min=0.0,
        soft_max=1.0,
        update=_sync_color_to_rgb,
    )


class Hd2ColorGraphItem(PropertyGroup):
    label: StringProperty(name="Label", default="")
    offset: IntProperty(name="Offset", default=0)
    points: CollectionProperty(type=Hd2ColorPoint)


class HD2_UL_Graphs(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        layout.label(text=item.label)


class HD2_UL_ColorGraphs(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        fields = []
        if getattr(data, "show_time_color", True):
            fields.append("time")
        if getattr(data, "show_color_color", True):
            fields.append("color")
        if not fields:
            fields = ["time", "color"]
        row_op = row.operator("hd2.particle_row_select", text=item.label, emboss=False)
        row_op.group = "color"
        row_op.graph_index = index
        row_op.fields = ",".join(fields)
        row_op.points_count = len(item.points)
        for i, point in enumerate(item.points):
            col = row.column(align=True)
            if getattr(data, "show_time_color", True):
                split_t = col.split(factor=0.65, align=True)
                split_t.prop(point, "x", text="")
                key = f"color:{index}:{i}:time"
                selected = key in _parse_selected_cells(data)
                op = split_t.operator("hd2.particle_cell_select", text="", depress=selected, icon="CHECKBOX_HLT" if selected else "CHECKBOX_DEHLT")
                op.key = key
            if getattr(data, "show_color_color", True):
                split = col.split(factor=0.65, align=True)
                split.prop(point, "color", text="")
                keyc = f"color:{index}:{i}:color"
                selectedc = keyc in _parse_selected_cells(data)
                opc = split.operator("hd2.particle_cell_select", text="", depress=selectedc, icon="CHECKBOX_HLT" if selectedc else "CHECKBOX_DEHLT")
                opc.key = keyc


class _HD2_UL_GraphsBase(UIList):
    graph_kind = ""

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        show_time = True
        show_value = True
        if self.graph_kind == "OPACITY":
            show_time = getattr(data, "show_time_opacity", True)
            show_value = getattr(data, "show_value_opacity", True)
        elif self.graph_kind == "SCALE":
            show_time = getattr(data, "show_time_intensity", True)
            show_value = getattr(data, "show_value_intensity", True)
        fields = []
        if show_time:
            fields.append("time")
        if show_value:
            fields.append("value")
        if not fields:
            fields = ["time", "value"]
        row_op = row.operator("hd2.particle_row_select", text=item.label, emboss=False)
        row_op.group = "graph"
        row_op.graph_index = index
        row_op.fields = ",".join(fields)
        row_op.points_count = len(item.points)
        for i, point in enumerate(item.points):
            col = row.column(align=True)
            if show_time:
                split_t = col.split(factor=0.65, align=True)
                split_t.prop(point, "x", text="")
                key = f"graph:{index}:{i}:time"
                selected = key in _parse_selected_cells(data)
                op = split_t.operator("hd2.particle_cell_select", text="", depress=selected, icon="CHECKBOX_HLT" if selected else "CHECKBOX_DEHLT")
                op.key = key
            if show_value:
                split_v = col.split(factor=0.65, align=True)
                split_v.prop(point, "y", text="")
                keyv = f"graph:{index}:{i}:value"
                selectedv = keyv in _parse_selected_cells(data)
                opv = split_v.operator("hd2.particle_cell_select", text="", depress=selectedv, icon="CHECKBOX_HLT" if selectedv else "CHECKBOX_DEHLT")
                opv.key = keyv

    def filter_items(self, context, data, propname):
        items = getattr(data, propname)
        flt_flags = []
        flt_neworder = []
        for item in items:
            show = item.kind == self.graph_kind
            flt_flags.append(self.bitflag_filter_item if show else 0)
        return flt_flags, flt_neworder


class HD2_UL_OpacityGraphs(_HD2_UL_GraphsBase):
    graph_kind = "OPACITY"


class HD2_UL_ScaleGraphs(_HD2_UL_GraphsBase):
    graph_kind = "SCALE"


class HD2_UL_OtherGraphs(_HD2_UL_GraphsBase):
    graph_kind = "OTHER"
#endregion

#region Emitters
class Hd2EmitterRatePoint(PropertyGroup):
    x: FloatProperty(name="X", default=0.0)
    y: FloatProperty(name="Y", default=0.0)


class Hd2EmitterBurstPoint(PropertyGroup):
    time: FloatProperty(name="Time", default=0.0)
    num_a: IntProperty(name="A", default=0)
    num_b: IntProperty(name="B", default=0)


class Hd2EmitterItem(PropertyGroup):
    label: StringProperty(name="Label", default="")
    system_index: IntProperty(name="System", default=0)
    offset: IntProperty(name="Offset", default=0)
    emitter_type: EnumProperty(
        name="Type",
        items=[
            ("RATE", "Rate", ""),
            ("BURST", "Burst", ""),
        ],
        default="RATE",
    )
    initial_rate_min: FloatProperty(name="Initial Rate Min", default=0.0)
    initial_rate_max: FloatProperty(name="Initial Rate Max", default=0.0)
    rate_points: CollectionProperty(type=Hd2EmitterRatePoint)
    burst_points: CollectionProperty(type=Hd2EmitterBurstPoint)


class HD2_UL_Emitters(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        layout.label(text=item.label)
#endregion

#region Transforms
class Hd2SystemTransformItem(PropertyGroup):
    label: StringProperty(name="Label", default="")
    system_index: IntProperty(name="System", default=0)
    rotation_offset: IntProperty(name="Rotation Offset", default=0)
    position_offset: IntProperty(name="Position Offset", default=0)
    rot_x: FloatProperty(name="Rot X", default=0.0, subtype="ANGLE")
    rot_y: FloatProperty(name="Rot Y", default=0.0, subtype="ANGLE")
    rot_z: FloatProperty(name="Rot Z", default=0.0, subtype="ANGLE")
    pos_x: FloatProperty(name="Pos X", default=0.0)
    pos_y: FloatProperty(name="Pos Y", default=0.0)
    pos_z: FloatProperty(name="Pos Z", default=0.0)


class HD2_UL_SystemTransforms(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        layout.label(text=item.label)
#endregion

#region Visualizers
class Hd2VisualizerItem(PropertyGroup):
    label: StringProperty(name="Label", default="")
    system_index: IntProperty(name="System", default=0)
    offset: IntProperty(name="Offset", default=0)
    vis_type: EnumProperty(
        name="Type",
        items=[
            ("BILLBOARD", "Billboard", ""),
            ("LIGHT", "Light", ""),
            ("MESH", "Mesh", ""),
            ("UNKNOWN3", "Unknown3", ""),
            ("UNKNOWN4", "Unknown4", ""),
        ],
        default="BILLBOARD",
    )
    unk1: IntProperty(name="Unknown 1", default=0)
    unk2: IntProperty(name="Unknown 2", default=0)
    material_id: StringProperty(name="Material ID", default="")
    unit_id: StringProperty(name="Unit ID", default="")
    mesh_id: StringProperty(name="Mesh ID", default="")


class HD2_UL_Visualizers(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        layout.label(text=item.label)
#endregion

#region Loaded Particles
class Hd2LoadedParticleItem(PropertyGroup):
    key: StringProperty(name="Key", default="")
    label: StringProperty(name="Label", default="")
    file_id: StringProperty(name="File ID", default="")
    is_archive: BoolProperty(name="Is Archive", default=False)


class HD2_UL_LoadedParticles(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.label(text=item.label)
        if item.is_archive:
            row.label(text=item.file_id)


class HD2_UL_ArchiveParticles(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        settings = context.scene.Hd2ParticleModderSettings
        is_active = bool(settings.is_archive and str(settings.entry_file_id) == str(item.item_name))
        op = row.operator(
            "helldiver2.particle_modder_edit",
            text=item.item_filter_name if item.item_filter_name else item.item_name,
            emboss=True,
            depress=is_active,
        )
        op.object_id = item.item_name
        row.operator("helldiver2.particle_search_used_ids", icon="VIEWZOOM", text="").object_id = item.item_name


class HD2_UL_DumpParticles(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        op = row.operator(
            "hd2.particle_loaded_select",
            text=item.file_id if item.file_id else item.label,
            emboss=True,
            depress=(getattr(data, "filepath", "") == item.key),
        )
        op.key = item.key
#endregion

#region Operators: Auto Edit Particle Selection
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
    is_archive: BoolProperty(name="Is Archive", default=False)
    entry_file_id: StringProperty(name="Entry File ID", default="")
    entry_type_id: StringProperty(name="Entry Type ID", default="")
    variables: CollectionProperty(type=Hd2ParticleVariableItem)
    variables_index: IntProperty(name="Variable Index", default=0)
    graphs: CollectionProperty(type=Hd2GraphItem)
    graphs_index: IntProperty(name="Graph Index", default=0, update=_on_graph_index_change)
    color_graphs: CollectionProperty(type=Hd2ColorGraphItem)
    color_graphs_index: IntProperty(name="Color Graph Index", default=0)
    color_point_index: IntProperty(name="Color Point Index", default=0)
    color_selected_indices: StringProperty(name="Color Selected", default="")
    selected_cells: StringProperty(name="Selected Cells", default="")
    last_selected_cell: StringProperty(name="Last Selected Cell", default="")
    color_apply: FloatVectorProperty(name="Color", size=3, subtype="COLOR", default=(1.0, 1.0, 1.0), min=0.0, max=1.0, soft_min=0.0, soft_max=1.0)
    emitters: CollectionProperty(type=Hd2EmitterItem)
    emitters_index: IntProperty(name="Emitter Index", default=0)
    transforms: CollectionProperty(type=Hd2SystemTransformItem)
    transforms_index: IntProperty(name="Transform Index", default=0)
    overall_position_x: FloatProperty(name="Overall Position X", default=0.0)
    overall_position_y: FloatProperty(name="Overall Position Y", default=0.0)
    overall_position_z: FloatProperty(name="Overall Position Z", default=0.0)
    overall_rotation_x: FloatProperty(name="Overall Rotation X", default=0.0, subtype="ANGLE")
    overall_rotation_y: FloatProperty(name="Overall Rotation Y", default=0.0, subtype="ANGLE")
    overall_rotation_z: FloatProperty(name="Overall Rotation Z", default=0.0, subtype="ANGLE")
    visualizers: CollectionProperty(type=Hd2VisualizerItem)
    visualizers_index: IntProperty(name="Visualizer Index", default=0)
    loaded_particles: CollectionProperty(type=Hd2LoadedParticleItem)
    loaded_particles_index: IntProperty(name="Loaded Particle Index", default=0, update=_on_loaded_particle_index_change)
    loaded_dump_particles: CollectionProperty(type=Hd2LoadedParticleItem)
    loaded_dump_particles_index: IntProperty(name="Loaded Dump Particle Index", default=-1, update=_on_loaded_dump_particle_index_change)
    particle_source_tab: EnumProperty(
        name="Particle Source",
        items=[
            ("ARCHIVE", "Archive Particles", ""),
            ("DUMP", "Dump Particles", ""),
        ],
        default="ARCHIVE",
    )
    ui_tab: EnumProperty(
        name="Tab",
        items=[
            ("COLOR", "Color", ""),
            ("OPACITY", "Opacity", ""),
            ("INTENSITY", "Intensity", ""),
            ("LIFETIME", "Lifetime", ""),
            ("VISUALIZERS", "Visualizers", ""),
            ("EMITTERS", "Emitters", ""),
            ("TRANSFORMS", "Transforms", ""),
            ("PARTICLES", "Particles", ""),
        ],
        default="COLOR",
    )
    show_time_color: BoolProperty(name="Show Time (Color)", default=True)
    show_color_color: BoolProperty(name="Show Color (Color)", default=True)
    show_time_opacity: BoolProperty(name="Show Time (Opacity)", default=True)
    show_value_opacity: BoolProperty(name="Show Opacity (Opacity)", default=True)
    show_time_intensity: BoolProperty(name="Show Time (Intensity)", default=True)
    show_value_intensity: BoolProperty(name="Show Size (Intensity)", default=True)
    color_preset_1: StringProperty(name="Color Preset 1", default="")
    color_preset_2: StringProperty(name="Color Preset 2", default="")
#endregion

#region Operators: Color Presets
class HD2_OT_ColorPresetSave(Operator):
    bl_idname = "hd2.particle_color_preset_save"
    bl_label = "Save Color Preset"
    bl_options = {"REGISTER", "UNDO"}

    slot: IntProperty(default=1)

    def execute(self, context):
        settings = context.scene.Hd2ParticleModderSettings
        if not settings.color_graphs:
            self.report({"ERROR"}, "No color graph loaded")
            return {"CANCELLED"}
        idx = settings.color_graphs_index
        if idx < 0 or idx >= len(settings.color_graphs):
            self.report({"ERROR"}, "Invalid color graph selection")
            return {"CANCELLED"}
        graph = settings.color_graphs[idx]
        payload = []
        for point in graph.points:
            payload.append([point.x, point.r, point.g, point.b])
        data = json.dumps(payload)
        if self.slot == 1:
            settings.color_preset_1 = data
        else:
            settings.color_preset_2 = data
        return {"FINISHED"}


class HD2_OT_ColorPresetLoad(Operator):
    bl_idname = "hd2.particle_color_preset_load"
    bl_label = "Load Color Preset"
    bl_options = {"REGISTER", "UNDO"}

    slot: IntProperty(default=1)

    def execute(self, context):
        settings = context.scene.Hd2ParticleModderSettings
        if not settings.color_graphs:
            self.report({"ERROR"}, "No color graph loaded")
            return {"CANCELLED"}
        idx = settings.color_graphs_index
        if idx < 0 or idx >= len(settings.color_graphs):
            self.report({"ERROR"}, "Invalid color graph selection")
            return {"CANCELLED"}
        data = settings.color_preset_1 if self.slot == 1 else settings.color_preset_2
        if not data:
            self.report({"ERROR"}, "Preset is empty")
            return {"CANCELLED"}
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            self.report({"ERROR"}, "Preset data is invalid")
            return {"CANCELLED"}
        graph = settings.color_graphs[idx]
        for i, point in enumerate(graph.points):
            if i >= len(payload):
                break
            x, r, g, b = payload[i]
            point.x = x
            point.r = r
            point.g = g
            point.b = b
            point.color = (
                (r / 255.0) * 1000.0,
                (g / 255.0) * 1000.0,
                (b / 255.0) * 1000.0,
            )
        return {"FINISHED"}
#endregion

#region Operators: Curve Mapping
class HD2_OT_GraphCurveReload(Operator):
    bl_idname = "hd2.particle_graph_curve_reload"
    bl_label = "Reload Curve"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.Hd2ParticleModderSettings
        if not settings.graphs:
            return {"CANCELLED"}
        idx = settings.graphs_index
        if idx < 0 or idx >= len(settings.graphs):
            return {"CANCELLED"}
        ok = _load_curve_from_graph(settings, settings.graphs[idx])
        if ok is False:
            self.report({"ERROR"}, "CurveMapping is not available in this Blender API context")
            return {"CANCELLED"}
        return {"FINISHED"}


class HD2_OT_GraphCurveApply(Operator):
    bl_idname = "hd2.particle_graph_curve_apply"
    bl_label = "Apply Curve"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.Hd2ParticleModderSettings
        if not settings.graphs:
            return {"CANCELLED"}
        idx = settings.graphs_index
        if idx < 0 or idx >= len(settings.graphs):
            return {"CANCELLED"}
        if STATE.graph_curve is None:
            self.report({"ERROR"}, "CurveMapping is not available in this Blender API context")
            return {"CANCELLED"}
        _sync_graph_from_curve(settings, settings.graphs[idx])
        return {"FINISHED"}
#endregion

#region Operators: Graph Editor
class HD2_OT_GraphEditor(Operator):
    bl_idname = "hd2.particle_graph_editor"
    bl_label = "Open Graph Editor"
    bl_options = {"REGISTER"}

    def modal(self, context, event):
        if event.type in {"ESC", "RIGHTMOUSE"}:
            self._finish(context)
            return {"CANCELLED"}

        settings = context.scene.Hd2ParticleModderSettings
        if not settings.graphs:
            self._finish(context)
            return {"CANCELLED"}

        if event.type == "LEFTMOUSE":
            if event.value == "PRESS":
                idx = settings.graphs_index
                if idx < 0 or idx >= len(settings.graphs):
                    return {"CANCELLED"}
                graph = settings.graphs[idx]
                coords = _graph_points_to_screen(graph.points, STATE.graph_rect)
                mx, my = event.mouse_region_x, event.mouse_region_y
                nearest = -1
                min_dist = 10.0
                for i, (x, y) in enumerate(coords):
                    d = ((mx - x) ** 2 + (my - y) ** 2) ** 0.5
                    if d < min_dist:
                        min_dist = d
                        nearest = i
                STATE.graph_drag_index = nearest
            elif event.value == "RELEASE":
                STATE.graph_drag_index = -1

        if event.type == "MOUSEMOVE" and STATE.graph_drag_index >= 0:
            idx = settings.graphs_index
            graph = settings.graphs[idx]
            mx, my = event.mouse_region_x, event.mouse_region_y
            x, y = _graph_screen_to_point(mx, my, STATE.graph_rect)
            graph.points[STATE.graph_drag_index].x = x
            graph.points[STATE.graph_drag_index].y = y
            context.area.tag_redraw()

        return {"RUNNING_MODAL"}

    def _finish(self, context):
        if STATE.graph_draw_handler is not None:
            bpy.types.SpaceView3D.draw_handler_remove(STATE.graph_draw_handler, "WINDOW")
            STATE.graph_draw_handler = None
        STATE.graph_modal_running = False
        context.area.tag_redraw()

    def execute(self, context):
        if STATE.graph_modal_running:
            return {"CANCELLED"}
        STATE.graph_draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            _draw_graph_callback, (), "WINDOW", "POST_PIXEL"
        )
        STATE.graph_modal_running = True
        context.window_manager.modal_handler_add(self)
        context.area.tag_redraw()
        return {"RUNNING_MODAL"}
#endregion

#region Operators: Loaded Particles
class HD2_OT_LoadedParticleSelect(Operator):
    bl_idname = "hd2.particle_loaded_select"
    bl_label = "Select Loaded Particle"
    bl_options = {"REGISTER", "UNDO"}

    key: StringProperty()

    def execute(self, context):
        settings = context.scene.Hd2ParticleModderSettings
        if self.key not in STATE.loaded_cache:
            if os.path.isfile(self.key):
                try:
                    with open(self.key, "rb") as f:
                        data = bytearray(f.read())
                except OSError as exc:
                    self.report({"ERROR"}, f"Failed to read file: {exc}")
                    return {"CANCELLED"}
                ok, err = load_from_bytes(context, data, self.key, 0, 0, False)
                if not ok:
                    self.report({"ERROR"}, err)
                    return {"CANCELLED"}
                return {"FINISHED"}
            self.report({"ERROR"}, "Particle not found in cache")
            return {"CANCELLED"}
        _cache_current(settings)
        entry = STATE.loaded_cache[self.key]
        ok, err = load_from_bytes(
            context,
            bytearray(entry["data"]),
            entry["label"],
            entry.get("file_id", ""),
            entry.get("type_id", ""),
            entry.get("is_archive", False),
            cache_current=False,
        )
        if not ok:
            self.report({"ERROR"}, err)
            return {"CANCELLED"}
        try:
            STATE.suspend_selection_sync = True
            for i, item in enumerate(settings.loaded_particles):
                if item.key == self.key:
                    settings.loaded_particles_index = i
                    break
            found_dump = False
            for i, item in enumerate(settings.loaded_dump_particles):
                if item.key == self.key:
                    settings.loaded_dump_particles_index = i
                    found_dump = True
                    break
            if not found_dump:
                settings.loaded_dump_particles_index = -1
            if not entry.get("is_archive", False):
                archive_index_id = f"index_{ParticleID}"
                if hasattr(context.scene, archive_index_id):
                    setattr(context.scene, archive_index_id, -1)
        finally:
            STATE.suspend_selection_sync = False
        return {"FINISHED"}
#endregion

#region Operators: Tabs
class HD2_OT_SetParticleTab(Operator):
    bl_idname = "hd2.particle_tab_set"
    bl_label = "Set Particle Tab"
    bl_options = {"REGISTER", "UNDO"}

    tab: StringProperty()

    def execute(self, context):
        settings = context.scene.Hd2ParticleModderSettings
        settings.ui_tab = self.tab
        return {"FINISHED"}
#endregion

#region Operators: Color Picker
class HD2_OT_ColorPointSelect(Operator):
    bl_idname = "hd2.particle_color_point_select"
    bl_label = "Select Color Point"
    bl_options = {"REGISTER", "UNDO"}

    index: IntProperty(default=0)
    toggle: BoolProperty(default=False)

    def execute(self, context):
        settings = context.scene.Hd2ParticleModderSettings
        settings.color_point_index = self.index
        indices = []
        if settings.color_selected_indices:
            try:
                indices = [int(v) for v in settings.color_selected_indices.split(",") if v != ""]
            except ValueError:
                indices = []
        if self.toggle:
            if self.index in indices:
                indices = [v for v in indices if v != self.index]
            else:
                indices.append(self.index)
        else:
            indices = [self.index]
        indices = sorted(set(indices))
        settings.color_selected_indices = ",".join(str(v) for v in indices)
        return {"FINISHED"}


#endregion

#region Operators: Color Selection
class HD2_OT_ColorSelectAll(Operator):
    bl_idname = "hd2.particle_color_select_all"
    bl_label = "Select All Colors"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.Hd2ParticleModderSettings
        if not settings.color_graphs:
            return {"CANCELLED"}
        cells = []
        for gidx, graph in enumerate(settings.color_graphs):
            for pidx in range(len(graph.points)):
                cells.append(f"color:{gidx}:{pidx}:color")
                cells.append(f"color:{gidx}:{pidx}:time")
        _set_selected_cells(settings, cells)
        return {"FINISHED"}


class HD2_OT_ColorSelectNone(Operator):
    bl_idname = "hd2.particle_color_select_none"
    bl_label = "Select None Colors"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.Hd2ParticleModderSettings
        settings.selected_cells = ""
        return {"FINISHED"}
#endregion

#region Operators: Apply Color
class HD2_OT_ColorApplySelected(Operator):
    bl_idname = "hd2.particle_color_apply_selected"
    bl_label = "Apply Color To Selected"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.Hd2ParticleModderSettings
        if not settings.color_graphs:
            return {"CANCELLED"}
        selection_map = _selected_color_cells_map_from_cells(_parse_selected_cells_value(settings.selected_cells))
        if not selection_map:
            return {"CANCELLED"}
        r = settings.color_apply[0] * 255.0
        g = settings.color_apply[1] * 255.0
        b = settings.color_apply[2] * 255.0
        # Apply to current settings
        for gidx, points in selection_map.items():
            if gidx < 0 or gidx >= len(settings.color_graphs):
                continue
            graph = settings.color_graphs[gidx]
            for pidx in points:
                if pidx < 0 or pidx >= len(graph.points):
                    continue
                point = graph.points[pidx]
                point.color = settings.color_apply
                point.r = r
                point.g = g
                point.b = b
        # Apply to all loaded cache entries
        for key, entry in STATE.loaded_cache.items():
            try:
                if key == STATE.filepath:
                    entry_cells = settings.selected_cells
                else:
                    entry_cells = entry.get("selected_cells", "")
                entry_selection_map = _selected_color_cells_map_from_cells(
                    _parse_selected_cells_value(entry_cells)
                )
                if not entry_selection_map:
                    continue
                data = bytearray(entry["data"])
                header = _parse_header(data)
                if header is None:
                    continue
                _apply_color_to_bytes(
                    data,
                    header["version"],
                    header["num_variables"],
                    header["num_systems"],
                    entry_selection_map,
                    (r, g, b),
                )
                entry["data"] = data
            except Exception:
                continue
        return {"FINISHED"}
#endregion

#region Menus: Presets
class HD2_MT_ColorSave(Menu):
    bl_label = "Save"

    def draw(self, context):
        layout = self.layout
        op = layout.operator("hd2.particle_color_preset_save", text="Save P1")
        op.slot = 1
        op = layout.operator("hd2.particle_color_preset_save", text="Save P2")
        op.slot = 2


class HD2_MT_ColorLoad(Menu):
    bl_label = "Load"

    def draw(self, context):
        layout = self.layout
        op = layout.operator("hd2.particle_color_preset_load", text="Load P1")
        op.slot = 1
        op = layout.operator("hd2.particle_color_preset_load", text="Load P2")
        op.slot = 2
#endregion


#region Operators: Cell Selection/Edit
def _parse_selected_cells(settings):
    return _parse_selected_cells_value(settings.selected_cells)


def _set_selected_cells(settings, cells):
    settings.selected_cells = "|".join(cells)


class HD2_OT_CellSelect(Operator):
    bl_idname = "hd2.particle_cell_select"
    bl_label = "Select Cell"
    bl_options = {"REGISTER", "UNDO"}

    key: StringProperty()
    toggle: BoolProperty(default=True)

    def invoke(self, context, event):
        self._shift = event.shift
        self._ctrl = event.ctrl
        self._alt = event.alt
        return self.execute(context)

    def execute(self, context):
        settings = context.scene.Hd2ParticleModderSettings
        cells = _parse_selected_cells(settings)
        key = self.key
        if self._shift and settings.last_selected_cell:
            # range select within same group (rectangle across graph index + point index)
            try:
                g1, gi1, pi1, f1 = settings.last_selected_cell.split(":")
                g2, gi2, pi2, f2 = key.split(":")
                if g1 == g2:
                    gi_start = min(int(gi1), int(gi2))
                    gi_end = max(int(gi1), int(gi2))
                    pi_start = min(int(pi1), int(pi2))
                    pi_end = max(int(pi1), int(pi2))
                    if g1 == "color":
                        fields = [f1] if f1 == f2 else ["time", "color"]
                    else:
                        fields = [f1] if f1 == f2 else ["time", "value"]
                    for gi in range(gi_start, gi_end + 1):
                        for pi in range(pi_start, pi_end + 1):
                            for field in fields:
                                cells.append(f"{g1}:{gi}:{pi}:{field}")
                    _set_selected_cells(settings, cells)
                    if not self._alt:
                        settings.last_selected_cell = key
                    return {"FINISHED"}
                else:
                    cells.append(key)
            except Exception:
                cells.append(key)
        elif self._ctrl:
            if key in cells:
                cells = [c for c in cells if c != key]
            else:
                cells.append(key)
        else:
            if key in cells:
                cells = [c for c in cells if c != key]
            else:
                cells.append(key)
        _set_selected_cells(settings, cells)
        if not self._alt:
            settings.last_selected_cell = key
        return {"FINISHED"}


class HD2_OT_RowSelect(Operator):
    bl_idname = "hd2.particle_row_select"
    bl_label = "Select Row"
    bl_options = {"REGISTER", "UNDO"}

    group: StringProperty()
    graph_index: IntProperty()
    fields: StringProperty(default="time,value")
    points_count: IntProperty(default=_GRAPH_POINTS)

    def execute(self, context):
        settings = context.scene.Hd2ParticleModderSettings
        cells = _parse_selected_cells(settings)
        field_list = [f.strip() for f in self.fields.split(",") if f.strip()]
        row_keys = []
        for pidx in range(max(0, int(self.points_count))):
            for field in field_list:
                row_keys.append(f"{self.group}:{self.graph_index}:{pidx}:{field}")
        if not row_keys:
            return {"CANCELLED"}
        if all(k in cells for k in row_keys):
            cells = [c for c in cells if c not in row_keys]
        else:
            for key in row_keys:
                if key not in cells:
                    cells.append(key)
        _set_selected_cells(settings, cells)
        settings.last_selected_cell = row_keys[0]
        return {"FINISHED"}


#endregion

#region Operators: Color Picker (HDR)
class HD2_OT_ColorPick(Operator):
    bl_idname = "hd2.particle_color_pick"
    bl_label = "Pick Color"
    bl_options = {"REGISTER", "UNDO"}

    key: StringProperty()
    color: FloatVectorProperty(name="Color", size=3, subtype="COLOR")

    def invoke(self, context, event):
        settings = context.scene.Hd2ParticleModderSettings
        parts = self.key.split(":")
        if len(parts) < 4:
            return {"CANCELLED"}
        gidx = int(parts[1])
        pidx = int(parts[2])
        graph = settings.color_graphs[gidx]
        point = graph.points[pidx]
        self.color = point.color
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        settings = context.scene.Hd2ParticleModderSettings
        parts = self.key.split(":")
        if len(parts) < 4:
            return {"CANCELLED"}
        gidx = int(parts[1])
        pidx = int(parts[2])
        graph = settings.color_graphs[gidx]
        point = graph.points[pidx]
        point.color = self.color
        point.r = self.color[0] * 255.0
        point.g = self.color[1] * 255.0
        point.b = self.color[2] * 255.0
        return {"FINISHED"}

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

        ok, err = load_from_bytes(context, data, self.filepath, 0, 0, False)
        if not ok:
            self.report({"ERROR"}, err)
            return {"CANCELLED"}

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
        ok, err = apply_settings_to_state(context, upgrade_to_current=False)
        if not ok:
            self.report({"ERROR"}, err)
            return {"CANCELLED"}
        return {"FINISHED"}


class HD2_OT_ParticleModderSave(Operator, ExportHelper):
    bl_idname = "hd2.particle_modder_save"
    bl_label = "Export .particle"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".particle"
    filter_glob: StringProperty(default="*.particle", options={"HIDDEN"})

    def execute(self, context):
        settings = context.scene.Hd2ParticleModderSettings
        if not settings.has_data or STATE.data is None:
            self.report({"ERROR"}, "No particle file loaded")
            return {"CANCELLED"}
        ok, err = apply_settings_to_state(context, upgrade_to_current=True)
        if not ok:
            self.report({"ERROR"}, err)
            return {"CANCELLED"}

        export_path = self.filepath or settings.filepath or "export.particle"
        if not export_path.lower().endswith(".particle"):
            export_path = f"{export_path}.particle"

        try:
            with open(export_path, "wb") as f:
                f.write(STATE.data)
        except OSError as exc:
            self.report({"ERROR"}, f"Failed to write file: {exc}")
            return {"CANCELLED"}

        return {"FINISHED"}
#endregion

#region Operators: Transform Tools
class HD2_OT_TransformOffsetApplyAll(Operator):
    bl_idname = "hd2.particle_transform_offset_apply_all"
    bl_label = "Apply Overall Transform"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.Hd2ParticleModderSettings
        if not settings.transforms:
            self.report({"ERROR"}, "No particle transforms loaded")
            return {"CANCELLED"}

        dx = float(settings.overall_position_x)
        dy = float(settings.overall_position_y)
        dz = float(settings.overall_position_z)
        rx = float(settings.overall_rotation_x)
        ry = float(settings.overall_rotation_y)
        rz = float(settings.overall_rotation_z)

        if dx == 0.0 and dy == 0.0 and dz == 0.0 and rx == 0.0 and ry == 0.0 and rz == 0.0:
            return {"CANCELLED"}

        for transform in settings.transforms:
            transform.pos_x += dx
            transform.pos_y += dy
            transform.pos_z += dz
            transform.rot_x += rx
            transform.rot_y += ry
            transform.rot_z += rz

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

        header = layout.box()
        title_row = header.row(align=True)
        title_row.label(text="Particle Modder", icon="PARTICLES")
        title_row.separator(factor=0.5)
        title_row.label(text=settings.filepath if settings.filepath else "No file loaded")
        btn_row = header.row(align=True)
        btn_row.operator("hd2.particle_modder_load", icon="FILE_FOLDER", text="Open")
        btn_row.operator("hd2.particle_modder_save", icon="FILE_TICK", text="Export .particle")
        if settings.is_archive:
            btn_row.operator("helldiver2.particle_modder_apply_entry", icon="FILE_TICK", text="Apply")
        else:
            btn_row.operator("hd2.particle_modder_apply", icon="CHECKMARK", text="Apply")

        particle_box = layout.box()
        particle_box.label(text="Particle Sources", icon="FILE_FOLDER")
        source_tabs = particle_box.row(align=True)
        source_tabs.prop(settings, "particle_source_tab", expand=True)
        scene = context.scene
        if settings.particle_source_tab == "ARCHIVE":
            if hasattr(scene, "Hd2ToolPanelSettings"):
                row = particle_box.row()
                row.prop(scene.Hd2ToolPanelSettings, "SearchField", icon="VIEWZOOM", text="")
            list_id = f"list_{ParticleID}"
            index_id = f"index_{ParticleID}_dummy"
            if hasattr(scene, list_id) and hasattr(scene, index_id):
                particle_box.template_list("HD2_UL_ArchiveParticles", list_id, scene, list_id, scene, index_id, rows=6)
                if hasattr(scene, "Hd2ToolPanelSettings"):
                    row = particle_box.row()
                    row.prop(scene.Hd2ToolPanelSettings, "LoadedArchives", text="Archive")
            else:
                particle_box.label(text="Particle list not available. Load an archive first.")
        else:
            particle_box.template_list("HD2_UL_DumpParticles", "", settings, "loaded_dump_particles", settings, "loaded_dump_particles_index", rows=6)
            if not settings.loaded_dump_particles:
                particle_box.label(text="No dump particle loaded. Use Open to load .particle/.particles file.")

        col = layout.column()

        # Custom tab row (hide legacy PARTICLES)
        tabs = col.row(align=True)
        tab_labels = {
            "COLOR": "Color",
            "OPACITY": "Opacity",
            "INTENSITY": "Intensity",
            "LIFETIME": "Lifetime",
            "VISUALIZERS": "Visualizers",
            "EMITTERS": "Emitters",
            "TRANSFORMS": "Transforms",
        }
        for tab in ("COLOR", "OPACITY", "INTENSITY", "LIFETIME", "VISUALIZERS", "EMITTERS", "TRANSFORMS"):
            op = tabs.operator("hd2.particle_tab_set", text=tab_labels.get(tab, tab.title()), depress=(settings.ui_tab == tab))
            op.tab = tab

        if settings.ui_tab == "COLOR":
            box = col.box()
            sub = box.row(align=True)
            sub.label(text="Color", icon="COLOR")
            sub.separator()
            sub.label(text="10 keys")
            tool = box.row(align=True)
            tool.prop(settings, "show_time_color", text="Time")
            tool.prop(settings, "show_color_color", text="Color")
            tool.prop(settings, "color_apply", text="")
            tool.operator("hd2.particle_color_apply_selected", text="Apply")
            tool.operator("hd2.particle_color_select_all", text="All")
            tool.operator("hd2.particle_color_select_none", text="None")
            tool.menu("HD2_MT_ColorSave", text="Save")
            tool.menu("HD2_MT_ColorLoad", text="Load")
            if not settings.has_data:
                box.label(text="Load a particle to edit color graphs.")
            header = box.row(align=True)
            header.label(text="")
            for i in range(1, 11):
                if settings.show_time_color and not settings.show_color_color:
                    header.label(text=f"Time {i}")
                if settings.show_color_color:
                    header.label(text=f"Color {i}")
            box.template_list("HD2_UL_ColorGraphs", "", settings, "color_graphs", settings, "color_graphs_index", rows=8)

        elif settings.ui_tab == "OPACITY":
            box = col.box()
            sub = box.row(align=True)
            sub.label(text="Opacity", icon="MOD_OPACITY")
            sub.separator()
            sub.label(text="10 keys")
            tool = box.row(align=True)
            tool.prop(settings, "show_time_opacity", text="Time")
            tool.prop(settings, "show_value_opacity", text="Opacity")
            tool.operator("hd2.particle_graph_editor", text="Open Graph")
            if not settings.has_data:
                box.label(text="Load a particle to edit opacity graphs.")
            header = box.row(align=True)
            header.label(text="")
            for i in range(1, 11):
                if settings.show_time_opacity and not settings.show_value_opacity:
                    header.label(text=f"Time {i}")
                if settings.show_value_opacity:
                    header.label(text=f"Opacity {i}")
            box.template_list("HD2_UL_OpacityGraphs", "", settings, "graphs", settings, "graphs_index", rows=8)

        elif settings.ui_tab == "INTENSITY":
            box = col.box()
            sub = box.row(align=True)
            sub.label(text="Intensity", icon="LIGHT")
            sub.separator()
            sub.label(text="10 keys")
            tool = box.row(align=True)
            tool.prop(settings, "show_time_intensity", text="Time")
            tool.prop(settings, "show_value_intensity", text="Size")
            tool.operator("hd2.particle_graph_editor", text="Open Graph")
            if not settings.has_data:
                box.label(text="Load a particle to edit intensity graphs.")
            header = box.row(align=True)
            header.label(text="")
            for i in range(1, 11):
                if settings.show_time_intensity and not settings.show_value_intensity:
                    header.label(text=f"Time {i}")
                if settings.show_value_intensity:
                    header.label(text=f"Size {i}")
            box.template_list("HD2_UL_ScaleGraphs", "", settings, "graphs", settings, "graphs_index", rows=8)

        elif settings.ui_tab == "LIFETIME":
            box = col.box()
            sub = box.row(align=True)
            sub.label(text="Lifetime", icon="TIME")
            box.prop(settings, "min_lifetime")
            box.prop(settings, "max_lifetime")
            box.separator()
            sub = box.row(align=True)
            sub.label(text="Variables", icon="SORTBYEXT")
            row = box.row()
            row.template_list("HD2_UL_ParticleVariables", "", settings, "variables", settings, "variables_index", rows=4)
            if not settings.has_data:
                box.label(text="Load a particle to edit lifetime and variables.")
            if settings.variables and 0 <= settings.variables_index < len(settings.variables):
                var = settings.variables[settings.variables_index]
                box.prop(var, "name_hash")
                box.prop(var, "x")
                box.prop(var, "y")
                box.prop(var, "z")

        elif settings.ui_tab == "VISUALIZERS":
            box = col.box()
            sub = box.row(align=True)
            sub.label(text="Visualizers", icon="SHADING_RENDERED")
            row = box.row()
            row.template_list("HD2_UL_Visualizers", "", settings, "visualizers", settings, "visualizers_index", rows=6)
            if settings.visualizers and 0 <= settings.visualizers_index < len(settings.visualizers):
                vis = settings.visualizers[settings.visualizers_index]
                box.prop(vis, "vis_type")
                box.label(text=f"System: {vis.system_index}")
                if vis.vis_type in {"BILLBOARD", "UNKNOWN3"}:
                    box.prop(vis, "unk1")
                    box.prop(vis, "unk2")
                    box.prop(vis, "material_id")
                elif vis.vis_type == "UNKNOWN4":
                    box.prop(vis, "material_id")
                elif vis.vis_type == "MESH":
                    box.prop(vis, "unit_id")
                    box.prop(vis, "mesh_id")
                    box.prop(vis, "material_id")
                elif vis.vis_type == "LIGHT":
                    box.label(text="Light visualizer has no editable fields yet")
        elif settings.ui_tab == "EMITTERS":
            box = col.box()
            sub = box.row(align=True)
            sub.label(text="Emitters", icon="FORCE_FORCE")
            row = box.row()
            row.template_list("HD2_UL_Emitters", "", settings, "emitters", settings, "emitters_index", rows=6)
            if settings.emitters and 0 <= settings.emitters_index < len(settings.emitters):
                emitter = settings.emitters[settings.emitters_index]
                box.prop(emitter, "emitter_type")
                box.label(text=f"System: {emitter.system_index}")
                if emitter.emitter_type == "RATE":
                    box.prop(emitter, "initial_rate_min")
                    box.prop(emitter, "initial_rate_max")
                    for point in emitter.rate_points:
                        r = box.row(align=True)
                        r.prop(point, "x", text="Time")
                        r.prop(point, "y", text="Rate")
                else:
                    for point in emitter.burst_points:
                        r = box.row(align=True)
                        r.prop(point, "time", text="Time")
                        r.prop(point, "num_a", text="A")
                        r.prop(point, "num_b", text="B")
        elif settings.ui_tab == "TRANSFORMS":
            box = col.box()
            sub = box.row(align=True)
            sub.label(text="Emitter Offset / Rotation", icon="ORIENTATION_GIMBAL")
            overall = box.box()
            overall.label(text="Overall Transform")
            pos_row = overall.row(align=True)
            pos_row.label(text="Position")
            pos_row.prop(settings, "overall_position_x", text="X")
            pos_row.prop(settings, "overall_position_y", text="Y")
            pos_row.prop(settings, "overall_position_z", text="Z")
            rot_row = overall.row(align=True)
            rot_row.label(text="Rotation")
            rot_row.prop(settings, "overall_rotation_x", text="X")
            rot_row.prop(settings, "overall_rotation_y", text="Y")
            rot_row.prop(settings, "overall_rotation_z", text="Z")
            overall.operator("hd2.particle_transform_offset_apply_all", text="Apply To All Systems")
            row = box.row()
            row.template_list("HD2_UL_SystemTransforms", "", settings, "transforms", settings, "transforms_index", rows=6)
            if settings.transforms and 0 <= settings.transforms_index < len(settings.transforms):
                t = settings.transforms[settings.transforms_index]
                box.label(text=f"System: {t.system_index}")
                box.label(text=f"Offsets R:{t.rotation_offset} P:{t.position_offset}")
                r = box.row(align=True)
                r.prop(t, "pos_x", text="Pos X")
                r.prop(t, "pos_y", text="Pos Y")
                r.prop(t, "pos_z", text="Pos Z")
                r = box.row(align=True)
                r.prop(t, "rot_x", text="Rot X")
                r.prop(t, "rot_y", text="Rot Y")
                r.prop(t, "rot_z", text="Rot Z")
#endregion

CLASSES = (
    Hd2ParticleVariableItem,
    HD2_UL_ParticleVariables,
    Hd2GraphPoint,
    Hd2GraphItem,
    Hd2ColorPoint,
    Hd2ColorGraphItem,
    HD2_UL_Graphs,
    HD2_UL_ColorGraphs,
    HD2_UL_OpacityGraphs,
    HD2_UL_ScaleGraphs,
    HD2_UL_OtherGraphs,
    Hd2EmitterRatePoint,
    Hd2EmitterBurstPoint,
    Hd2EmitterItem,
    HD2_UL_Emitters,
    Hd2SystemTransformItem,
    HD2_UL_SystemTransforms,
    Hd2VisualizerItem,
    HD2_UL_Visualizers,
    Hd2LoadedParticleItem,
    HD2_UL_LoadedParticles,
    HD2_UL_ArchiveParticles,
    HD2_UL_DumpParticles,
    Hd2ParticleModderSettings,
    HD2_OT_ColorPresetSave,
    HD2_OT_ColorPresetLoad,
    HD2_OT_ColorPointSelect,
    HD2_OT_ColorPick,
    HD2_OT_ColorSelectAll,
    HD2_OT_ColorSelectNone,
    HD2_OT_ColorApplySelected,
    HD2_MT_ColorSave,
    HD2_MT_ColorLoad,
    HD2_OT_GraphEditor,
    HD2_OT_LoadedParticleSelect,
    HD2_OT_CellSelect,
    HD2_OT_RowSelect,
    HD2_OT_SetParticleTab,
    HD2_OT_ParticleModderLoad,
    HD2_OT_ParticleModderApply,
    HD2_OT_ParticleModderSave,
    HD2_OT_TransformOffsetApplyAll,
    HD2_PT_ParticleModder,
)


def register_properties():
    bpy.types.Scene.Hd2ParticleModderSettings = PointerProperty(type=Hd2ParticleModderSettings)


def unregister_properties():
    if hasattr(bpy.types.Scene, "Hd2ParticleModderSettings"):
        del bpy.types.Scene.Hd2ParticleModderSettings

#region Public API
def load_from_bytes(context, data, label, file_id=0, type_id=0, is_archive=False, cache_current=True):
    settings = context.scene.Hd2ParticleModderSettings
    try:
        if not isinstance(data, bytearray):
            data = bytearray(data)
    except Exception:
        return False, "Invalid particle data buffer"
    if cache_current:
        _cache_current(settings)
    cached_entry = STATE.loaded_cache.get(label)
    if cached_entry is not None and cache_current:
        data = bytearray(cached_entry["data"])
    try:
        effect = _parse_particle_effect_model(data)
    except Exception as exc:
        return False, f"Unsupported or invalid particle file: {exc}"

    STATE.filepath = label
    STATE.data = data
    STATE.version = effect.version

    settings.filepath = label
    settings.version = effect.version
    settings.min_lifetime = effect.min_lifetime
    settings.max_lifetime = effect.max_lifetime
    settings.num_variables = effect.num_variables
    settings.num_systems = effect.num_particle_systems
    settings.is_archive = bool(is_archive)
    settings.entry_file_id = str(file_id)
    settings.entry_type_id = str(type_id)
    settings.has_data = True

    cached = STATE.loaded_cache.get(label)
    if cached is not None:
        settings.selected_cells = cached.get("selected_cells", "")
        settings.last_selected_cell = cached.get("last_selected_cell", "")
    else:
        settings.selected_cells = ""
        settings.last_selected_cell = ""

    found = None
    found_index = -1
    for i, item in enumerate(settings.loaded_particles):
        if item.key == label:
            found = item
            found_index = i
            break
    display_id = str(file_id) if is_archive else _dump_particle_id_from_label(label)
    if found is None:
        item = settings.loaded_particles.add()
        item.key = label
        item.label = display_id
        item.file_id = display_id
        item.is_archive = bool(is_archive)
        try:
            STATE.suspend_selection_sync = True
            settings.loaded_particles_index = len(settings.loaded_particles) - 1
        finally:
            STATE.suspend_selection_sync = False
    else:
        found.label = display_id
        found.file_id = display_id
        found.is_archive = bool(is_archive)
        try:
            STATE.suspend_selection_sync = True
            settings.loaded_particles_index = found_index if found_index >= 0 else settings.loaded_particles_index
        finally:
            STATE.suspend_selection_sync = False

    dump_found = None
    dump_found_index = -1
    for i, item in enumerate(settings.loaded_dump_particles):
        if item.key == label:
            dump_found = item
            dump_found_index = i
            break
    if is_archive:
        if dump_found is not None:
            settings.loaded_dump_particles.remove(dump_found_index)
    else:
        if dump_found is None:
            ditem = settings.loaded_dump_particles.add()
            ditem.key = label
            ditem.label = display_id
            ditem.file_id = display_id
            ditem.is_archive = False
            try:
                STATE.suspend_selection_sync = True
                settings.loaded_dump_particles_index = len(settings.loaded_dump_particles) - 1
            finally:
                STATE.suspend_selection_sync = False
        else:
            dump_found.label = display_id
            dump_found.file_id = display_id
            dump_found.is_archive = False
            try:
                STATE.suspend_selection_sync = True
                settings.loaded_dump_particles_index = dump_found_index if dump_found_index >= 0 else settings.loaded_dump_particles_index
            finally:
                STATE.suspend_selection_sync = False
    try:
        STATE.suspend_selection_sync = True
        if is_archive:
            settings.loaded_dump_particles_index = -1
        else:
            archive_index_id = f"index_{ParticleID}"
            if hasattr(context.scene, archive_index_id):
                setattr(context.scene, archive_index_id, -1)
    finally:
        STATE.suspend_selection_sync = False

    settings.variables.clear()
    for variable in effect.variables:
        item = settings.variables.add()
        item.name_hash = str(variable.name_hash)
        item.x = float(variable.x)
        item.y = float(variable.y)
        item.z = float(variable.z)
    settings.variables_index = 0 if settings.variables else 0

    settings.graphs.clear()
    settings.color_graphs.clear()
    settings.transforms.clear()
    settings.emitters.clear()
    settings.visualizers.clear()

    for system_index, system in enumerate(effect.particle_systems):
        for i, graph in enumerate(system.other_graphs):
            item = settings.graphs.add()
            item.label = f"Other S{system_index} #{i}"
            item.kind = "OTHER"
            item.offset = int(system.offset + system.other_graph_offsets[i])
            item.repeat_count = 2
            _copy_graph_points_to_settings(item, graph)

        for i, graph in enumerate(system.scale_graphs):
            if graph is None:
                continue
            item = settings.graphs.add()
            item.label = f"Scale S{system_index} #{i}"
            item.kind = "SCALE"
            item.offset = int(system.offset + system.scale_graph_offsets[i])
            item.repeat_count = 2
            _copy_graph_points_to_settings(item, graph)

        for i, graph in enumerate(system.opacity_graphs):
            item = settings.graphs.add()
            item.label = f"Opacity S{system_index} #{i}"
            item.kind = "OPACITY"
            item.offset = int(system.offset + system.opacity_graph_offsets[i])
            item.repeat_count = 2
            _copy_graph_points_to_settings(item, graph)

        for i, graph in enumerate(system.color_graphs):
            item = settings.color_graphs.add()
            item.label = f"Color S{system_index} #{i}"
            item.offset = int(system.offset + system.color_graph_offsets[i])
            _copy_color_points_to_settings(item, graph)

        transform = settings.transforms.add()
        transform.label = f"S{system_index}"
        transform.system_index = system_index
        transform.rotation_offset = int(system.offset + 0x78)
        transform.position_offset = int(system.offset + 0xA8)
        transform.rot_x = float(system.rotation.rotation_euler[0])
        transform.rot_y = float(system.rotation.rotation_euler[1])
        transform.rot_z = float(system.rotation.rotation_euler[2])
        transform.pos_x = float(system.position.position[0])
        transform.pos_y = float(system.position.position[1])
        transform.pos_z = float(system.position.position[2])

        if system.visualizer is not None:
            item = settings.visualizers.add()
            item.label = f"V{len(settings.visualizers)-1} S{system_index}"
            item.system_index = system_index
            item.offset = int(system.offset + system.visualizer_offset)
            vis_type = system.visualizer.visualizer_type
            if vis_type == _VIS_BILLBOARD:
                item.vis_type = "BILLBOARD"
            elif vis_type == _VIS_LIGHT:
                item.vis_type = "LIGHT"
            elif vis_type == _VIS_MESH:
                item.vis_type = "MESH"
            elif vis_type == _VIS_UNKNOWN3:
                item.vis_type = "UNKNOWN3"
            elif vis_type == _VIS_UNKNOWN4:
                item.vis_type = "UNKNOWN4"
            item.unk1 = int(getattr(system.visualizer, "unk1", 0))
            item.unk2 = int(getattr(system.visualizer, "unk2", 0))
            item.material_id = str(getattr(system.visualizer, "material_id", 0))
            item.unit_id = str(getattr(system.visualizer, "unit_id", 0))
            item.mesh_id = str(getattr(system.visualizer, "mesh_id", 0))

    settings.graphs_index = 0 if settings.graphs else 0
    settings.color_graphs_index = 0 if settings.color_graphs else 0
    settings.transforms_index = 0 if settings.transforms else 0
    settings.emitters_index = 0
    settings.visualizers_index = 0 if settings.visualizers else 0
    STATE.loaded_cache[label] = {
        "label": label,
        "data": bytearray(data),
        "file_id": str(file_id),
        "type_id": str(type_id),
        "is_archive": bool(is_archive),
        "selected_cells": settings.selected_cells,
        "last_selected_cell": settings.last_selected_cell,
    }
    return True, ""


def _apply_settings_to_state_data(settings):
    if not settings.has_data or STATE.data is None:
        return False, "No particle data loaded"
    try:
        effect = _parse_particle_effect_model(bytearray(STATE.data))
    except Exception as exc:
        return False, f"Failed to parse particle data: {exc}"

    if settings.graphs and STATE.graph_curve is not None:
        idx = settings.graphs_index
        if 0 <= idx < len(settings.graphs):
            _sync_graph_from_curve(settings, settings.graphs[idx])

    effect.min_lifetime = float(settings.min_lifetime)
    effect.max_lifetime = float(settings.max_lifetime)
    effect.variables = []
    for var in settings.variables:
        variable = _ParticleEffectVariableModel()
        try:
            variable.name_hash = int(var.name_hash, 0)
        except (ValueError, TypeError):
            variable.name_hash = 0
        variable.x = float(var.x)
        variable.y = float(var.y)
        variable.z = float(var.z)
        effect.variables.append(variable)
    effect.num_variables = len(effect.variables)
    effect.num_particle_systems = len(effect.particle_systems)

    other_graphs = [g for g in settings.graphs if g.kind == "OTHER"]
    scale_graphs = [g for g in settings.graphs if g.kind == "SCALE"]
    opacity_graphs = [g for g in settings.graphs if g.kind == "OPACITY"]
    color_graphs = list(settings.color_graphs)
    transforms = list(settings.transforms)
    visualizers = list(settings.visualizers)

    other_index = 0
    scale_index = 0
    opacity_index = 0
    color_index = 0
    transform_index = 0
    visualizer_index = 0

    for system in effect.particle_systems:
        for graph in system.other_graphs:
            if other_index < len(other_graphs):
                _apply_settings_graph_to_model(other_graphs[other_index], graph)
                other_index += 1
        for graph in system.scale_graphs:
            if graph is None:
                continue
            if scale_index < len(scale_graphs):
                _apply_settings_graph_to_model(scale_graphs[scale_index], graph)
                scale_index += 1
        for graph in system.opacity_graphs:
            if opacity_index < len(opacity_graphs):
                _apply_settings_graph_to_model(opacity_graphs[opacity_index], graph)
                opacity_index += 1
        for graph in system.color_graphs:
            if color_index < len(color_graphs):
                _apply_settings_color_to_model(color_graphs[color_index], graph)
                color_index += 1
        if transform_index < len(transforms):
            transform = transforms[transform_index]
            system.position.position = [
                float(transform.pos_x),
                float(transform.pos_y),
                float(transform.pos_z),
            ]
            system.rotation.rotation_euler = (
                float(transform.rot_x),
                float(transform.rot_y),
                float(transform.rot_z),
            )
            transform_index += 1
        if system.visualizer is not None and visualizer_index < len(visualizers):
            vis = visualizers[visualizer_index]
            system.visualizer.visualizer_type = {
                "BILLBOARD": _VIS_BILLBOARD,
                "LIGHT": _VIS_LIGHT,
                "MESH": _VIS_MESH,
                "UNKNOWN3": _VIS_UNKNOWN3,
                "UNKNOWN4": _VIS_UNKNOWN4,
            }.get(vis.vis_type, _VIS_BILLBOARD)
            system.visualizer.unk1 = int(vis.unk1)
            system.visualizer.unk2 = int(vis.unk2)
            try:
                system.visualizer.material_id = int(vis.material_id, 0)
            except (ValueError, TypeError):
                system.visualizer.material_id = 0
            try:
                system.visualizer.unit_id = int(vis.unit_id, 0)
            except (ValueError, TypeError):
                system.visualizer.unit_id = 0
            try:
                system.visualizer.mesh_id = int(vis.mesh_id, 0)
            except (ValueError, TypeError):
                system.visualizer.mesh_id = 0
            visualizer_index += 1

    stream = _PMStream(bytearray(STATE.data))
    try:
        effect.write_to_memory_stream(stream)
    except Exception as exc:
        return False, f"Failed to write particle data: {exc}"

    STATE.data = bytearray(stream.data)
    STATE.version = effect.version
    settings.version = effect.version
    return True, ""


def _apply_settings_to_state_data_in_place(settings):
    if not settings.has_data or STATE.data is None:
        return False, "No particle data loaded"
    header = _parse_header(STATE.data)
    if header is None:
        return False, "Unsupported or invalid particle file"

    if settings.graphs and STATE.graph_curve is not None:
        idx = settings.graphs_index
        if 0 <= idx < len(settings.graphs):
            _sync_graph_from_curve(settings, settings.graphs[idx])

    data = bytearray(STATE.data)
    _apply_min_max(data, settings.min_lifetime, settings.max_lifetime)
    _write_variables(data, header["version"], settings.variables)

    graph_infos = _scan_graphs(
        data,
        header["version"],
        header["num_variables"],
        header["num_systems"],
    )
    other_graphs = [g for g in settings.graphs if g.kind == "OTHER"]
    scale_graphs = [g for g in settings.graphs if g.kind == "SCALE"]
    opacity_graphs = [g for g in settings.graphs if g.kind == "OPACITY"]
    color_graphs = list(settings.color_graphs)

    other_index = 0
    scale_index = 0
    opacity_index = 0
    color_index = 0
    for info in graph_infos:
        for offset in info.other_offsets:
            if other_index >= len(other_graphs):
                break
            _write_bytes_at(
                data,
                int(info.offset + offset),
                _pack_graph_from_settings(other_graphs[other_index]) * 2,
            )
            other_index += 1
        for i, offset in enumerate(info.scale_offsets):
            if i < len(info.scale_graphs) and info.scale_graphs[i] is None:
                continue
            if scale_index >= len(scale_graphs):
                break
            _write_bytes_at(
                data,
                int(info.offset + offset),
                _pack_graph_from_settings(scale_graphs[scale_index]) * 2,
            )
            scale_index += 1
        for offset in info.opacity_offsets:
            if opacity_index >= len(opacity_graphs):
                break
            _write_bytes_at(
                data,
                int(info.offset + offset),
                _pack_graph_from_settings(opacity_graphs[opacity_index]) * 2,
            )
            opacity_index += 1
        for offset in info.color_offsets:
            if color_index >= len(color_graphs):
                break
            _write_bytes_at(
                data,
                int(info.offset + offset),
                _pack_color_graph_from_settings(color_graphs[color_index]),
            )
            color_index += 1

    for transform in settings.transforms:
        _write_transform_to_bytes(data, transform)

    STATE.data = data
    STATE.version = header["version"]
    settings.version = header["version"]
    return True, ""


def apply_settings_to_state(context, upgrade_to_current=False):
    settings = context.scene.Hd2ParticleModderSettings
    if upgrade_to_current:
        ok, err = _apply_settings_to_state_data(settings)
    else:
        ok, err = _apply_settings_to_state_data_in_place(settings)
    if not ok:
        return False, err
    STATE.loaded_cache[settings.filepath] = {
        "label": settings.filepath,
        "data": bytearray(STATE.data),
        "file_id": settings.entry_file_id,
        "type_id": settings.entry_type_id,
        "is_archive": settings.is_archive,
        "selected_cells": settings.selected_cells,
        "last_selected_cell": settings.last_selected_cell,
    }
    ok, err = load_from_bytes(
        context,
        bytearray(STATE.data),
        settings.filepath,
        settings.entry_file_id,
        settings.entry_type_id,
        settings.is_archive,
        cache_current=False,
    )
    if not ok:
        return False, err
    return True, ""
#endregion

