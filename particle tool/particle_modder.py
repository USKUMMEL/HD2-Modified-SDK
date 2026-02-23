# ==============================================================================
# HD2 Particle Modder - Version 2.0.4
# A tool for modifying Helldivers 2 particle effect files
# ==============================================================================

# ==============================================================================
# IMPORTS
# ==============================================================================

# Standard library imports
import ast
import os
import time
import struct
from functools import partial
import xml.etree.cElementTree as ET

# Scientific computing imports
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import math
import numpy as np
from scipy.spatial.transform import Rotation

# PySide6 (Qt) imports
from PySide6.QtCore import Qt, QRect, QAbstractItemModel, Signal, QXmlStreamWriter, QXmlStreamReader, QItemSelectionModel
from PySide6.QtCharts import QLineSeries, QChart, QChartView, QValueAxis
from PySide6.QtGui import QStandardItem, QStandardItemModel, QPalette, QColor, QAction, QShortcut, QKeySequence, QIcon, QDoubleValidator, QValidator, QPen, QIntValidator
from PySide6.QtWidgets import QApplication, QMainWindow, QMenu, QHBoxLayout, QVBoxLayout, QScrollArea, QSizePolicy, \
    QWidget, QSplitter, QFileDialog, QTabWidget, QColorDialog, QTableView, QStyledItemDelegate, QStyle, QToolButton, QStatusBar, QLabel, QMessageBox, QFileSystemModel, QLineEdit, QTreeWidget, QTreeWidgetItem, QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsItem, QListWidget, QListWidgetItem, QFrame, QDialog, QCheckBox, QDialogButtonBox
from PySide6.QtGui import QUndoCommand, QUndoStack

# ==============================================================================
# CONSTANTS
# ==============================================================================

VERSION = "2.0.4"
CURRENT_PARTICLE_EFFECT_VERSION = 0x71
VALID_PARTICLE_EFFECT_VERSIONS = [0x71, 0x6F, 0x6E, 0x6D]

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def clear_layout(layout):
    if layout is not None:
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                clear_layout(item.layout())

# ==============================================================================
# CORE DATA CLASSES - Particle System Components
# ==============================================================================

class EmitterPosition:

    def __init__(self):
        self.fileOffset = 0
        self.position = [0, 0, 0]

    @classmethod
    def fromBytes(cls, data):
        g = EmitterPosition()
        g.position = list(struct.unpack("<fff", data[0:12]))
        return g
        
    def to_bytes(self):
        return struct.pack("<fff", *self.position)

    def setOffset(self, offset):
        self.fileOffset = offset

    def getOffset(self):
        return self.fileOffset

class EmitterRotation:

    def __init__(self):
        self.fileOffset = 0
        self.rotation = None

    @classmethod
    def fromBytes(cls, data):
        g = EmitterRotation()
        g.rotation = Rotation.from_matrix([
            list(struct.unpack("<fff", data[0:12])),
            list(struct.unpack("<fff", data[16:28])),
            list(struct.unpack("<fff", data[32:44]))
        ])
        return g
        
    def to_bytes(self):
        rot_mat = self.rotation.as_matrix()
        row1 = struct.pack("<fff", *rot_mat[0])
        row2 = struct.pack("<fff", *rot_mat[1])
        row3 = struct.pack("<fff", *rot_mat[2])
        zero_as_bytes = bytearray(4)
        return row1 + zero_as_bytes + row2 + zero_as_bytes + row3 + zero_as_bytes

    def getRotationMatrix(self):
        return self.rotation.as_matrix()

    def getQuaternion(self):
        return self.rotation.as_quat()

    def setOffset(self, offset):
        self.fileOffset = offset

    def getOffset(self):
        return self.fileOffset

# ==============================================================================
# PARTICLE SYSTEM COMPONENTS
# ==============================================================================

class Visualizer:
    
    BILLBOARD = 0
    LIGHT = 1
    MESH = 2
    UNKNOWN3 = 3
    UNKNOWN4 = 4
    
    def __init__(self):
        pass
    
    def from_memory_stream(self, stream):
        self.visualizer_type = stream.uint32_read()
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
            
    def write_to_memory_stream(self, stream):
        if self.visualizer_type == Visualizer.BILLBOARD:
            data = struct.pack("<IIIQ", self.visualizer_type, self.unk1, self.unk2, self.material_id) + self.data
            stream.write(data)
        elif self.visualizer_type == Visualizer.LIGHT:
            data = struct.pack("<I", self.visualizer_type) + self.data
            stream.write(data)
        elif self.visualizer_type == Visualizer.MESH:
            data = struct.pack("<IQQQ", self.visualizer_type, self.unit_id, self.mesh_id, self.material_id) + self.data
            stream.write(data)
        elif self.visualizer_type == Visualizer.UNKNOWN3:
            data = struct.pack("<IIIQ", self.visualizer_type, self.unk1, self.unk2, self.material_id) + self.data
            stream.write(data)
        elif self.visualizer_type == Visualizer.UNKNOWN4:
            data = struct.pack("<IQ", self.visualizer_type, self.material_id) + self.data
            stream.write(data)
        
class Graph:
    def __init__(self):
        pass
        
    def from_memory_stream(self, stream):
        self.x = [stream.float32_read() for _ in range(10)]
        self.y = [stream.float32_read() for _ in range(10)]
        
    def write_to_memory_stream(self, stream):
        stream.write(struct.pack("<ffffffffff", *self.x))
        stream.write(struct.pack("<ffffffffff", *self.y))
        
class ColorGraph:
    def __init__(self):
        pass
        
    def from_memory_stream(self, stream):
        self.x = [stream.float32_read() for _ in range(10)]
        self.y = [[stream.float32_read() for _ in range(3)] for _ in range(10)]
        
    def write_to_memory_stream(self, stream):
        stream.write(struct.pack("<ffffffffff", *self.x))
        for color in self.y:
            stream.write(struct.pack("<fff", *color))
            
class BurstEmitterGraph:
    
    def __init__(self):
        self.times = []
        self.num_particles = []
        
    def from_memory_stream(self, stream):
        for _ in range(10):
            self.times.append(stream.float32_read())
            self.num_particles.append((stream.uint32_read(), stream.uint32_read()))
        
    def write_to_memory_stream(self, stream):
        for i in range(10):
            stream.write(struct.pack("<fII", self.times[i], self.num_particles[i][0], self.num_particles[i][1]))

class Emitter:
    
    BURST = 0x0C
    RATE = 0x0B
    
    def __init__(self):
        pass
        
    def from_memory_stream(self, stream):
        self.emitter_type = stream.uint32_read()
        if self.emitter_type == Emitter.BURST:
            burst_graph = BurstEmitterGraph()
            burst_graph.from_memory_stream(stream)
            self.burst_graph = burst_graph
        elif self.emitter_type == Emitter.RATE:
            self.initial_rate_min = stream.float32_read()
            self.initial_rate_max = stream.float32_read()
            rate_graph = Graph()
            rate_graph.from_memory_stream(stream)
            self.rate_graph = rate_graph
            
    def write_to_memory_stream(self, stream):
        stream.advance(4)
        if self.emitter_type == Emitter.BURST:
            self.burst_graph.write_to_memory_stream(stream)
        elif self.emitter_type == Emitter.RATE:
            stream.write(struct.pack("<ff", self.initial_rate_min, self.initial_rate_max))
            self.rate_graph.write_to_memory_stream(stream)
        

class ParticleSystem:
    def __init__(self, version):
        self.version = version
        self.scale_graphs = []
        self.opacity_graphs = []
        self.color_graphs = []
        self.color_graph_offsets = []
        self.other_graph_offsets = []
        self.other_graphs = []
        self.emitter_offsets = []
        self.emitters = []
        self.visualizer = None
        self.offset = 0
        self.max_num_particles = 0
        self.component_chunk = bytearray()
        self.emitter_chunk = bytearray()
        
    def is_rendering(self):
        return self.non_rendering == 0
        
    def from_memory_stream(self, stream):
        self.scale_graphs.clear()
        self.opacity_graphs.clear()
        self.color_graphs.clear()
        self.color_graph_offsets.clear()
        self.other_graphs.clear()
        self.other_graph_offsets.clear()
        self.emitters.clear()
        self.offset = stream.tell()
        self.max_num_particles = stream.uint32_read()
        self.num_components = stream.uint32_read()
        self.unk1 = stream.read(68)
        #self.advance(4)
        #self.advance(64)
        self.non_rendering = stream.uint32_read()
        self.unk2 = stream.read(40)
        # rotation
        self.rotation = EmitterRotation.fromBytes(stream.read(48))
        # position
        self.position = EmitterPosition.fromBytes(stream.read(12))
        self.unk3 = stream.read(52)
        self.component_list_offset = stream.uint32_read()
        self.unk4 = stream.read(4)
        self.emitter_offset = stream.uint32_read()
        self.unk5 = stream.read(8)
        self.visualizer_offset = stream.uint32_read()
        self.size = stream.uint32_read()
        stream.seek(self.offset + self.component_list_offset)
        self.component_chunk = stream.read(self.emitter_offset - self.component_list_offset)
        stream.seek(self.offset + self.emitter_offset)
        self.emitter_chunk = stream.read(self.visualizer_offset-self.emitter_offset)
        if not self.is_rendering():
            stream.seek(self.offset + self.size)
            return
        if self.visualizer_offset == self.size:
            stream.seek(self.offset + self.size)
            return
        '''    
        # get emitters
        stream.seek(self.offset + self.emitter_offset)
        #emitter_type = stream.uint32_read()
        
        stop = False
        while not stop:
            emitter_type = stream.uint32_read()
            while emitter_type not in [Emitter.BURST, Emitter.RATE]:
                emitter_type = stream.uint32_read()
                if stream.tell() >= self.offset + self.visualizer_offset:
                    stop = True
                    break
            if not stop:
                stream.advance(-4)
                offset = stream.tell()
                emitter = Emitter()
                emitter.from_memory_stream(stream)
                if stream.tell() >= self.offset + self.visualizer_offset:
                    break
                self.emitters.append(emitter)
                self.emitter_offsets.append(offset)
        '''
            
        # get visualizer
        stream.seek(self.offset + self.visualizer_offset)
        visualizer = Visualizer()
        visualizer.from_memory_stream(stream)
        self.visualizer = visualizer
        
        # get graphs/components
        while stream.tell() < self.offset + self.size:
            # get component type
            component_type = stream.uint32_read()
            subtype = 0
            if component_type in [0x05, 0x04, 0x0F]: # graph, maybe. check subtype
                subtype = stream.uint32_read()
                if subtype < 0x20:
                    stream.advance(-4)
                    continue
                else:
                    stream.advance(-8)
            elif component_type == 0x00: # skip
                continue
            elif component_type == 0x11: # don't like this, but there doesn't seem to be a good way to handle this
                if stream.tell() + 284 < self.offset + self.size:
                    stream.advance(284)
            elif component_type == 0x0B:
                stream.advance(24)
                continue
            else: # skip
                continue
            if stream.tell() + 16 > self.offset + self.size:
                break
            component_type = [stream.uint32_read() for _ in range(4)]
            if component_type[0] == 0x04 and component_type[1] >= 0x20: # graph
                stream.advance(4)
                self.other_graph_offsets.append(stream.tell() - self.offset)
                unk_graph = Graph()
                unk_graph.from_memory_stream(stream)
                unk_graph.from_memory_stream(stream)
                self.other_graphs.append(unk_graph)
                stream.advance(8) # unknown data
            elif component_type[0] == 0x05 and component_type[1] >= 0x20: # color graph
                # color graph
                stream.advance(-4)
                self.color_graph_offsets.append(stream.tell() - self.offset)
                scale = Graph()
                scale.from_memory_stream(stream)
                scale.from_memory_stream(stream)
                self.scale_graphs.append(scale)
                opacity = Graph()
                opacity.from_memory_stream(stream)
                opacity.from_memory_stream(stream)
                self.opacity_graphs.append(opacity)
                color = ColorGraph()
                color.from_memory_stream(stream)
                self.color_graphs.append(color)
                stream.advance(16) # unknown data
            elif component_type[1] == 0x05 and component_type[2] >= 0x20: # color graph
                self.color_graph_offsets.append(stream.tell() - self.offset)
                scale = Graph()
                scale.from_memory_stream(stream)
                scale.from_memory_stream(stream)
                self.scale_graphs.append(scale)
                opacity = Graph()
                opacity.from_memory_stream(stream)
                opacity.from_memory_stream(stream)
                self.opacity_graphs.append(opacity)
                color = ColorGraph()
                color.from_memory_stream(stream)
                self.color_graphs.append(color)  
                stream.advance(16)
            elif component_type[0] == 0x0F and component_type[1] >= 0x20: # color graph, no scale
                stream.advance(-4)
                self.color_graph_offsets.append(stream.tell() - self.offset)
                scale = None
                #scale.from_memory_stream(stream)
                #scale.from_memory_stream(stream)
                self.scale_graphs.append(scale)
                opacity = Graph()
                opacity.from_memory_stream(stream)
                opacity.from_memory_stream(stream)
                self.opacity_graphs.append(opacity)
                color = ColorGraph()
                color.from_memory_stream(stream)
                self.color_graphs.append(color)
                stream.advance(16)
            elif component_type[0] == 0x0B: # some float data
                stream.advance(12)
            else:
                continue
            
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
        self.visualizer.write_to_memory_stream(stream)
        for index, offset in enumerate(self.color_graph_offsets):
            stream.seek(offset + self.offset)
            if self.scale_graphs[index] is not None:
                self.scale_graphs[index].write_to_memory_stream(stream)
                self.scale_graphs[index].write_to_memory_stream(stream)
            self.opacity_graphs[index].write_to_memory_stream(stream)
            self.opacity_graphs[index].write_to_memory_stream(stream)
            self.color_graphs[index].write_to_memory_stream(stream)
        for index, offset in enumerate(self.other_graph_offsets):
            stream.seek(offset + self.offset)
            self.other_graphs[index].write_to_memory_stream(stream)
            self.other_graphs[index].write_to_memory_stream(stream)
        
        
class ParticleEffectVariable:
    def __init__(self):
        self.name_hash = 0
        self.x = 0
        self.y = 0
        self.z = 0

# ==============================================================================
# MAIN PARTICLE EFFECT CLASS
# ==============================================================================

CURRENT_PARTICLE_EFFECT_VERSION = 0x71

class ParticleEffect:
    def __init__(self):
        self.variables = []
        self.particle_systems = []
        self.min_lifetime = 0
        self.max_lifetime = 0
        self.num_variables = 0
        self.num_particle_systems = 0
        self.version = 0
        
    def from_memory_stream(self, stream):
        self.variables.clear()
        self.particle_systems.clear()
        self.version = stream.uint32_read()
        if self.version not in VALID_PARTICLE_EFFECT_VERSIONS:
            return
        self.min_lifetime = stream.float32_read()
        self.max_lifetime = stream.float32_read()
        stream.advance(8)
        self.num_variables = stream.uint32_read()
        self.num_particle_systems = stream.uint32_read()
        stream.advance(44)
        if self.version in [0x6F, 0x71]:
            stream.advance(8)
        for _ in range(self.num_variables):
            new_var = ParticleEffectVariable()
            new_var.name_hash = stream.uint32_read()
            self.variables.append(new_var)
        for variable in self.variables:
            variable.x = stream.float32_read()
            variable.y = stream.float32_read()
            variable.z = stream.float32_read()
        for _ in range(self.num_particle_systems):
            new_system = ParticleSystem(self.version)
            new_system.from_memory_stream(stream)
            self.particle_systems.append(new_system)
            
    def write_to_memory_stream(self, stream):
        stream.seek(0)
        stream.write(struct.pack("<I", CURRENT_PARTICLE_EFFECT_VERSION))
        stream.write(struct.pack("<ff", self.min_lifetime, self.max_lifetime))
        stream.advance(8)
        stream.write(self.num_variables.to_bytes(4, byteorder="little"))
        stream.write(self.num_particle_systems.to_bytes(4, byteorder="little"))
        if self.version in [0x6F, 0x71]:
            stream.advance(52)
        else: # insert 8 bytes to match version 0x6F
            stream.advance(44)
            stream.data[stream.tell():stream.tell()] = bytearray(8)
            stream.advance(8)
            for particle_system in self.particle_systems:
                particle_system.offset += 8
        for variable in self.variables:
            stream.write(struct.pack("<I", variable.name_hash))
        for variable in self.variables:
            stream.write(struct.pack("<fff", variable.x, variable.y, variable.z))
        for particle_system in self.particle_systems:
            stream.seek(particle_system.offset)
            particle_system.write_to_memory_stream(stream)
        if self.version != 0x71 and len(self.particle_systems) > 0:
            updated_offset = 0
            for particle_system in self.particle_systems:
                if particle_system.is_rendering():
                    offset = particle_system.offset + updated_offset + particle_system.emitter_offset - 16
                    stream.seek(offset)
                    value = stream.uint32_read()
                    stream.seek(stream.tell()+12)
                    if value == 8: # option 1. Insert 0x38 at the start. Find where 08 00 00 00 00 00 00 00 20 00 00 00 00 is and insert 0x30, 0x34, 0x38
                        stream.seek(particle_system.offset + updated_offset + 0xFC)
                        stream.write(struct.pack("<II", particle_system.visualizer_offset+16, particle_system.size+16))
                        stream.seek(particle_system.offset + updated_offset + particle_system.emitter_offset + 8)
                        stream.data[stream.tell():stream.tell()] = b'\xFF\xFF\xFF\xFF'
                        
                        replace_offset = stream.data.find(b'\x08\x00\x00\x00\x00\x00\x00\x00', particle_system.offset + particle_system.emitter_offset + updated_offset, particle_system.offset + particle_system.visualizer_offset + updated_offset)
                        if replace_offset == -1:
                            print("Error updating particle effect")
                            return 1
                        stream.seek(replace_offset + 8)
                        replace_value = stream.uint32_read()
                        stream.data[replace_offset+8:replace_offset+8] = struct.pack("<III", replace_value+0x10, replace_value+0x14, replace_value+0x18)
                        updated_offset += 16
                    else: # option 2: insert 0xFFFFFFFF at the start.
                        stream.seek(particle_system.offset + updated_offset + 0xFC)
                        stream.write(struct.pack("<II", particle_system.visualizer_offset+4, particle_system.size+4))
                        stream.seek(particle_system.offset + updated_offset + particle_system.emitter_offset + 8)
                        stream.data[stream.tell():stream.tell()] = b'\xFF\xFF\xFF\xFF'
                        updated_offset += 4
        if self.version != 0x71:
            stream.seek(0)
            self.from_memory_stream(stream)
            self.version = 0x71

# ==============================================================================
# MEMORY STREAM UTILITY
# ==============================================================================

class MemoryStream:
    '''
    Modified from https://github.com/kboykboy2/io_scene_helldivers2 with permission from kboykboy
    '''
    def __init__(self, Data=b"", io_mode = "read"):
        self.location = 0
        self.data = bytearray(Data)
        self.io_mode = io_mode
        self.endian = "<"

    def open(self, Data, io_mode = "read"): # Open Stream
        self.data = bytearray(Data)
        self.io_mode = io_mode

    def set_read_mode(self):
        self.io_mode = "read"

    def set_write_mode(self):
        self.io_mode = "write"

    def is_reading(self):
        return self.io_mode == "read"

    def is_writing(self):
        return self.io_mode == "write"

    def seek(self, location): # Go To Position In Stream
        self.location = location
        if self.location > len(self.data):
            missing_bytes = self.location - len(self.data)
            self.data += bytearray(missing_bytes)

    def tell(self): # Get Position In Stream
        return self.location
        
    def insert(self, length):
        self.data[self.location:self.location] = bytearray(length)
        
    def delete(self, length):
        self.data[self.location:self.location+length] = b''

    def read(self, length=-1): # read Bytes From Stream
        if length == -1:
            length = len(self.data) - self.location
        if self.location + length > len(self.data):
            raise Exception("reading past end of stream")

        newData = self.data[self.location:self.location+length]
        self.location += length
        return bytearray(newData)

    def advance(self, offset):
        self.location += offset
        if self.location < 0:
            self.location = 0
        if self.location > len(self.data):
            missing_bytes = self.location - len(self.data)
            self.data += bytearray(missing_bytes)

    def write(self, bytes): # Write Bytes To Stream
        length = len(bytes)
        if self.location + length > len(self.data):
            missing_bytes = (self.location + length) - len(self.data)
            self.data += bytearray(missing_bytes)
        self.data[self.location:self.location+length] = bytearray(bytes)
        self.location += length

    def read_format(self, format, size):
        format = self.endian+format
        return struct.unpack(format, self.read(size))[0]

    def bytes(self, value, size = -1):
        if size == -1:
            size = len(value)
        if len(value) != size:
            value = bytearray(size)

        if self.is_reading():
            return bytearray(self.read(size))
        elif self.is_writing():
            self.write(value)
            return bytearray(value)
        return value

    def int8_read(self):
        return self.read_format('b', 1)

    def uint8_read(self):
        return self.read_format('B', 1)

    def int16_read(self):
        return self.read_format('h', 2)

    def uint16_read(self):
        return self.read_format('H', 2)

    def int32_read(self):
        return self.read_format('i', 4)

    def uint32_read(self):
        return self.read_format('I', 4)

    def int64_read(self):
        return self.read_format('q', 8)

    def uint64_read(self):
        return self.read_format('Q', 8)
        
    def float32_read(self):
        return self.read_format('f', 4)

# ==============================================================================
# UI VIEW COMPONENTS
# ==============================================================================

class EmitterView(QWidget):
    
    def __init__(self, emitters: list[Emitter], parent=None):
        super().__init__(parent)
        
        self.emitters = emitters
        
        self.layout = QVBoxLayout()
        
        for emitter in sorted(self.emitters, key=lambda e: e.emitter_type):
            if emitter.emitter_type == Emitter.RATE:
                chart = QChart()
                chart.setTitle("Rate over time")
                series = QLineSeries()
                series.setName("Rate")
                pen = QPen(0xFF0000)
                pen.setWidth(5)
                series.setPen(pen)
                for x, y in zip(emitter.rate_graph.x, emitter.rate_graph.y):
                    if not x == 10000:
                        series.append(x, y)
                chart.addSeries(series)
                chart.legend().hide()
                chart.createDefaultAxes()
                chart.axes(Qt.Orientation.Horizontal)[0].setRange(0, 1)
                chart.axes(Qt.Orientation.Horizontal)[0].setTickCount(2)
                chart.axes(Qt.Orientation.Vertical)[0].setRange(0, 1)
                chart.axes(Qt.Orientation.Vertical)[0].setTickCount(2)
                chartView = QChartView(chart)
                chartView.setFixedWidth(200)
                chartView.setFixedHeight(200)
                self.emitterLayout = QHBoxLayout()
                self.emitterLayout2 = QVBoxLayout()
                emitterLabel = QLabel(f"Rate Emitter", self)
                self.emitterLayout.addWidget(emitterLabel)
                emitterLabelMin = QLabel("Min: ", self)
                emitterLabelMax = QLabel("Max: ", self)
                emitterEditMin = QLineEdit(self)
                emitterEditMin.setFixedWidth(130)
                emitterEditMin.setText(str(emitter.initial_rate_min))
                emitterEditMin.setValidator(QDoubleValidator())
                emitterEditMax = QLineEdit(self)
                emitterEditMax.setFixedWidth(130)
                emitterEditMax.setText(str(emitter.initial_rate_max))
                emitterEditMax.setValidator(QDoubleValidator())
                self.emitterLayout.addWidget(emitterLabelMin)
                self.emitterLayout.addWidget(emitterEditMin)
                self.emitterLayout.addWidget(emitterLabelMax)
                self.emitterLayout.addWidget(emitterEditMax)
                self.emitterLayout.addStretch(1)
                self.emitterLayout2.addLayout(self.emitterLayout)
                self.emitterLayout2.addWidget(chartView)
                self.layout.addLayout(self.emitterLayout2)
            elif emitter.emitter_type == Emitter.BURST:
                self.emitterLayout = QVBoxLayout()
                emitterLabel = QLabel(f"Burst Emitter", self)
                self.emitterLayout.addWidget(emitterLabel)
                self.titleLayout = QHBoxLayout()
                self.titleLayout.addWidget(QLabel("Time"))
                self.titleLayout.addWidget(QLabel("Min Burst"))
                self.titleLayout.addWidget(QLabel("Max Burst"))
                self.emitterLayout.addLayout(self.titleLayout)
                for time, value in zip(emitter.burst_graph.times, emitter.burst_graph.num_particles):
                    min = value[0]
                    max = value[1]
                    self.rowLayout = QHBoxLayout()
                    timeEdit = QLineEdit()
                    timeEdit.setText(str(time))
                    timeEdit.setValidator(QDoubleValidator())
                    self.rowLayout.addWidget(timeEdit)
                    minEdit = QLineEdit()
                    minEdit.setText(str(min))
                    minEdit.setValidator(QIntValidator())
                    self.rowLayout.addWidget(minEdit)
                    maxEdit = QLineEdit()
                    maxEdit.setText(str(max))
                    maxEdit.setValidator(QIntValidator())
                    self.rowLayout.addWidget(maxEdit)
                    self.emitterLayout.addLayout(self.rowLayout)
                self.layout.addLayout(self.emitterLayout)
        self.layout.addStretch(1)
        self.setLayout(self.layout)
        
        # def setData
        
class ParticleSystemView(QWidget):
    
    '''
    Container for showing a particle system. Includes visualizer, color graph, opacity graph, scale graph, rotation, and position
    '''
    
    def __init__(self, particleSystem, trailSpawner = -1, parent=None):
        super().__init__(parent)
        self.particleSystem = particleSystem
        
        self.layout = QVBoxLayout()
        
        self.emitterView = EmitterView(self.particleSystem.emitters)
        
        if trailSpawner == -1:
            self.visualizerView = VisualizerView(self.particleSystem.visualizer)
            self.layout.addWidget(self.visualizerView)
            
        else:
            self.trailSpawnerLabel = QLabel(f"Trail Spawner for Particle System {trailSpawner}", parent=self)
            self.layout.addWidget(self.trailSpawnerLabel)
            
        self.layout.addWidget(self.emitterView)
        
        for graph in self.particleSystem.other_graphs:
            g = GraphView(graph)
            self.layout.addWidget(g)
        
        self.layout.addStretch(1)
        self.setLayout(self.layout)
        
class ParticleEffectView(QWidget):
    '''
    Container for showing a particle effect. Includes tab widget for each particle system plus info about the overall effect (lifetime)
    '''
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        
        # particle effect data
        lifetimeWidth = 100
        self.lifetimeWidget = QWidget(self)
        self.lifetimeLayout = QHBoxLayout()
        self.lifetimeLabel1 = QLabel("Lifetime: ", self)
        self.lifetimeLabel2 = QLabel(" - ", self)
        self.lifetimeLabel3 = QLabel(" seconds", self)
        self.lifetimeValidator = QDoubleValidator()
        self.lifetimeMinEdit = QLineEdit(self)
        self.lifetimeMinEdit.setValidator(self.lifetimeValidator)
        self.lifetimeMinEdit.editingFinished.connect(self.setLifetime)
        self.lifetimeMinEdit.setFixedWidth(lifetimeWidth)
        self.lifetimeMaxEdit = QLineEdit(self)
        self.lifetimeMaxEdit.setValidator(self.lifetimeValidator)
        self.lifetimeMaxEdit.editingFinished.connect(self.setLifetime)
        self.lifetimeMaxEdit.setFixedWidth(lifetimeWidth)
        self.lifetimeLayout.addWidget(self.lifetimeLabel1)
        self.lifetimeLayout.addWidget(self.lifetimeMinEdit)
        self.lifetimeLayout.addWidget(self.lifetimeLabel2)
        self.lifetimeLayout.addWidget(self.lifetimeMaxEdit)
        self.lifetimeLayout.addWidget(self.lifetimeLabel3)
        self.lifetimeLayout.addStretch(1)
        self.lifetimeWidget.setLayout(self.lifetimeLayout)
        
        self.layout.addWidget(self.lifetimeWidget)
        
        # tabs for the particle systems
        self.tabWidget = QTabWidget(self)
                
        self.layout.addWidget(self.tabWidget)
                
        self.setLayout(self.layout)
        
    def loadData(self, particleEffect):
        self.particleEffect = particleEffect
        
        self.lifetimeMinEdit.setText(str(self.particleEffect.min_lifetime))
        self.lifetimeMaxEdit.setText(str(self.particleEffect.max_lifetime))
        
        self.tabWidget.clear()
        count = 0
        for particleSystem in self.particleEffect.particle_systems:
            if particleSystem.is_rendering():
                if particleSystem.visualizer_offset != particleSystem.size:
                    particleSystemView = ParticleSystemView(particleSystem)
                    self.tabWidget.addTab(particleSystemView, f"Particle System {count}")
                else:
                    particleSystemView = ParticleSystemView(particleSystem, trailSpawner=count+1)
                    self.tabWidget.addTab(particleSystemView, f"Particle System {count}")
                count += 1
        
    def setLifetime(self):
        self.particleEffect.min_lifetime = float(self.lifetimeMinEdit.text())
        self.particleEffect.max_lifetime = float(self.lifetimeMaxEdit.text())

class BigIntValidator(QDoubleValidator):
    def __init__(self, bottom=float('-inf'), top=float('inf')):
        super(BigIntValidator, self).__init__(bottom, top, 0)
        self.setNotation(QDoubleValidator.StandardNotation)

    def validate(self, text, pos):
        if text.endswith('.'):
            return QValidator.Invalid, text, pos
        return super(BigIntValidator, self).validate(text, pos)
        
class VisualizerView(QWidget):
    
    def __init__(self, visualizer, parent=None):
        super().__init__(parent)
        lineWidth = 160
        self.int32Max = (2**32)-1
        self.int64Max = (2**64)-1
        self.layout = QVBoxLayout()
        self.visualizer = visualizer
        self.materialIdLabel = self.unitIdLabel=self.meshIdLabel = None
        self.materialIdEdit = self.unitIdEdit = self.meshIdEdit = None
        self.visualizerLabel = QLabel("", parent=self)
        
        self.int32Validator = BigIntValidator(0, self.int32Max)
        self.int64Validator = BigIntValidator(0, self.int64Max)
        if self.visualizer.visualizer_type == Visualizer.BILLBOARD:
            # material ID
            self.materialIdLabel = QLabel(f"Material: ", parent=self)
            self.materialIdEdit = QLineEdit(f"{self.visualizer.material_id}", parent=self)
            self.materialIdEdit.setFixedWidth(lineWidth)
            self.materialIdEdit.editingFinished.connect(self.materialIdChanged)
            self.materialIdEdit.setValidator(self.int64Validator)
            self.visualizerLabel.setText("Visualizer Type: Billboard")
        elif self.visualizer.visualizer_type == Visualizer.LIGHT:
            # no IDs
            self.visualizerLabel.setText("Visualizer Type: Light")
        elif self.visualizer.visualizer_type == Visualizer.MESH:
            # material, unit, and mesh ID
            self.materialIdEdit = QLineEdit(f"{self.visualizer.material_id}", parent=self)
            self.materialIdEdit.setValidator(self.int64Validator)
            self.materialIdEdit.editingFinished.connect(self.materialIdChanged)
            self.materialIdEdit.setFixedWidth(lineWidth)
            self.materialIdLabel = QLabel(f"Material: ", parent=self)
            
            self.unitIdEdit = QLineEdit(f"{self.visualizer.unit_id}", parent=self)
            self.unitIdEdit.setValidator(self.int64Validator)
            self.unitIdEdit.editingFinished.connect(self.unitIdChanged)
            self.unitIdEdit.setFixedWidth(lineWidth)
            self.unitIdLabel = QLabel(f"Unit: ", parent=self)
            
            self.meshIdEdit = QLineEdit(f"{self.visualizer.mesh_id}", parent=self)
            self.meshIdEdit.editingFinished.connect(self.meshIdChanged)
            self.meshIdEdit.setValidator(self.int64Validator)
            self.meshIdEdit.setFixedWidth(lineWidth)
            self.meshIdLabel = QLabel(f"Mesh: ", parent=self)
            
            self.visualizerLabel.setText("Visualizer Type: Mesh")
            
        elif self.visualizer.visualizer_type == Visualizer.UNKNOWN3:
            self.materialIdLabel = QLabel(f"Material: ", parent=self)
            self.materialIdEdit = QLineEdit(f"{self.visualizer.material_id}", parent=self)
            self.materialIdEdit.setFixedWidth(lineWidth)
            self.materialIdEdit.editingFinished.connect(self.materialIdChanged)
            self.materialIdEdit.setValidator(self.int64Validator)
            self.visualizerLabel.setText("Visualizer Type: UNKNOWN")
        elif self.visualizer.visualizer_type == Visualizer.UNKNOWN4:
            self.materialIdLabel = QLabel(f"Material: ", parent=self)
            self.materialIdEdit = QLineEdit(f"{self.visualizer.material_id}", parent=self)
            self.materialIdEdit.setFixedWidth(lineWidth)
            self.materialIdEdit.editingFinished.connect(self.materialIdChanged)
            self.materialIdEdit.setValidator(self.int64Validator)
            self.visualizerLabel.setText("Visualizer Type: UNKNOWN")
            
        self.layout.addWidget(self.visualizerLabel, alignment=Qt.AlignTop | Qt.AlignLeft)
        for label, edit in zip([self.materialIdLabel, self.unitIdLabel, self.meshIdLabel], [self.materialIdEdit, self.unitIdEdit, self.meshIdEdit]):
            if label is not None and edit is not None:
                layout = QHBoxLayout()
                layout.addWidget(label)
                layout.addWidget(edit)
                layout.addStretch(1)
                self.layout.addLayout(layout)
        self.layout.addStretch(1)
            
        self.setLayout(self.layout)
            
    def meshIdChanged(self):
        newId = int(self.meshIdEdit.text())
        self.visualizer.mesh_id = newId & self.int32Max
        
    def materialIdChanged(self):
        newId = int(self.materialIdEdit.text())
        self.visualizer.material_id = newId & self.int64Max
        
    def unitIdChanged(self):
        newId = int(self.unitIdEdit.text())
        self.visualizer.unit_id = newId & self.int64Max
        
class ParticleMaterialView(QWidget):
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.layout2 = QVBoxLayout()
        self.setLayout(self.layout)
        
    def loadData(self, particleEffect):
        clear_layout(self.layout)
        clear_layout(self.layout2)
        scrollArea = QScrollArea()
        #scrollArea.setBackgroundRole(QPalette.Dark)
        self.particleEffect = particleEffect
        count = 0
        for particleSystem in self.particleEffect.particle_systems:
            if particleSystem.is_rendering():
                if particleSystem.visualizer_offset != particleSystem.size:
                    label = QLabel(f"Particle System {count}")
                    visualizerView = VisualizerView(particleSystem.visualizer)
                    self.layout.addWidget(label)
                    self.layout.addWidget(visualizerView)
                else:
                    pass
                count += 1
        container = QWidget()
        container.setLayout(self.layout)
        scrollArea.setWidget(container)
        self.layout2.addWidget(scrollArea)
        self.setLayout(self.layout2)
        
    def setLifetime(self):
        self.particleEffect.min_lifetime = float(self.lifetimeMinEdit.text())
        self.particleEffect.max_lifetime = float(self.lifetimeMaxEdit.text())

class MovablePoint(QGraphicsEllipseItem):
    def __init__(self, x, y, radius=10):
        super().__init__(x - radius, y - radius, 2 * radius, 2 * radius)
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable) #
        self.setBrush(Qt.GlobalColor.blue)

    def mouseMoveEvent(self, event):
        # Update connected lines here if needed
        # For instance, if you have a list of lines associated with this point
        # You'd iterate through them and update their start or end points
        super().mouseMoveEvent(event) # Ensure default behavior is called
        

def graphs_set_dark_mode():
    plt.style.use("dark_background")
    mpl.rcParams['axes.prop_cycle'] = cycler(color=['#ffd500'])
    mpl.rcParams['figure.facecolor'] = "#333333"
    mpl.rcParams['axes.facecolor'] = "#333333"

def graphs_set_light_mode():
    plt.style.use("default")

# ==============================================================================
# GRAPH VISUALIZATION COMPONENTS
# ==============================================================================

class GraphWidget(QWidget):
    
    def __init__(self):
        super().__init__()
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.x = np.arange(0, 1, 1.0)
        self.y = np.arange(0, 1, 1.0)
        self.xscale = 1
        self.yscale = 1
        self.line, = self.ax.plot(self.x, self.y, marker="o")
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.grabbed_point = False
        self.index = 0
        self.layout = QHBoxLayout()
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)
        self.canvas.mpl_connect('button_press_event', self.onclick)
        self.canvas.mpl_connect('button_release_event', self.onrelease)
        self.canvas.mpl_connect('motion_notify_event', self.onmove)
        self.margin = 0.2
        self.ax.grid()
        self.fig.canvas.draw()
        
    def set_xlabel(self, label):
        self.ax.set_xlabel(label)
        
    def set_ylabel(self, label):
        self.ax.set_ylabel(label)
        
    def set_title(self, title):
        self.ax.set_title(title)
        
    def set_axis_format(self, axis, format):
        if axis not in ["x", "y"]:
            raise ValueError(f"Unknown axis {axis}")
        if format not in ["decimal", "percent"]:
            raise ValueError(f"Unknown axis format {format}")
            
        if axis == "x":
            if format == "percent":
                self.ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0))
            elif format == "decimal":
                self.ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        elif axis == "y":
            if format == "percent":
                self.ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0))
            elif format == "decimal":
                self.ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        
    def set_data(self, xdata, ydata):
        self.x = [i for i in xdata if i != 10000.0]
        self.y = ydata[:len(self.x)]
        axis_min = min(self.x)
        axis_max = max(self.x)
        self.xscale = (axis_max-axis_min)
        if self.xscale == 0:
            self.xscale = 1
        self.ax.set_xlim(axis_min-(self.xscale*self.margin), axis_max+(self.xscale*self.margin))
        axis_min = min(ydata)
        axis_max = max(ydata)
        self.yscale = (axis_max-axis_min)
        if self.yscale == 0:
            self.yscale = 1
        self.ax.set_ylim(axis_min-(self.yscale*self.margin), axis_max+(self.yscale*self.margin))
        self.line.set_xdata(self.x)
        self.line.set_ydata(self.y)
        self.fig.canvas.draw()
        
    def get_data(self):
        return self.x, self.y
        
    def onclick(self, event):
        min_distance = 999999999999
        for i, point in enumerate(self.x):
            distance = math.sqrt(abs((point-event.xdata)/(self.xscale/5))**2 + abs((self.y[i]-event.ydata)/(self.yscale/5))**2)
            if distance < min_distance:
                min_distance = distance
                self.index = i
        print(min_distance)
        if event.button == mpl.backend_bases.MouseButton.LEFT:
            if min_distance < 0.2:
                self.grabbed_point = True
            else:
                if len(self.x) == 10:
                    return
                for i, point in enumerate(self.x):
                    if event.xdata > point:
                        self.index = i
                self.x = np.insert(self.x, self.index+1, event.xdata)
                self.y = np.insert(self.y, self.index+1, event.ydata)
                self.line.set_xdata(self.x)
                self.line.set_ydata(self.y)
                self.fig.canvas.draw()
        elif event.button == mpl.backend_bases.MouseButton.RIGHT:
            if len(self.x) == 1:
                return
            if min_distance < 0.5:
                self.x = np.delete(self.x, self.index)
                self.y = np.delete(self.y, self.index)
                self.line.set_xdata(self.x)
                self.line.set_ydata(self.y)
                self.fig.canvas.draw()
        
    def onrelease(self, event):
        self.grabbed_point = False
        axis_min = min(self.x)
        axis_max = max(self.x)
        self.xscale = (axis_max-axis_min)
        if self.xscale == 0:
            self.xscale = 1
        self.ax.set_xlim(axis_min-(self.xscale*self.margin), axis_max+(self.xscale*self.margin))
        axis_min = min(self.y)
        axis_max = max(self.y)
        self.yscale = (axis_max-axis_min)
        if self.yscale == 0:
            self.yscale = 1
        self.ax.set_ylim(axis_min-(self.yscale*self.margin), axis_max+(self.yscale*self.margin))
        self.fig.canvas.draw()
        
    def onmove(self, event):
        if self.grabbed_point:
            self.x[self.index] = event.xdata
            self.y[self.index] = event.ydata
            self.line.set_xdata(self.x)
            self.line.set_ydata(self.y)
            self.fig.canvas.draw()
       
class GraphView(QWidget):
    def __init__(self, graph):
        super().__init__()
        self.layout = QVBoxLayout()
        self.graph = graph
        #self.resize(400, 400)
        
        # process graph
        self.load_graph(graph)
        
        self.setLayout(self.layout)
        
    def load_graph(self, graph):
        self.graphWidget = GraphWidget()
        self.graph = graph
        self.graphWidget.set_data(graph.x, graph.y)
        self.layout.addWidget(self.graphWidget)
        self.setLayout(self.layout)

# ==============================================================================
# UNDO/REDO SUPPORT
# ==============================================================================

class LifetimeView(QWidget):
    def __init__(self, particleEffect):
        pass
        

class SetDataCommand(QUndoCommand):
    def __init__(self, model, index, new_value, description="Edit Cell"):
        super().__init__(description)
        self.model = model
        self.index = index
        self.row = index.row()
        self.column = index.column()
        self.new_value = new_value
        self.old_value = index.data()

    def undo(self):
        self.model.blockSignals(True)
        self.model.setData(self.model.index(self.row, self.column), self.old_value)
        self.model.blockSignals(False)

    def redo(self):
        self.model.blockSignals(True)
        self.model.setData(self.model.index(self.row, self.column), self.new_value)
        self.model.blockSignals(False)

class OpacityGradient:

    def __init__(self):
        self.fileOffset = 0
        self.opacities = []

    @classmethod
    def fromBytes(cls, data):
        g = OpacityGradient()
        for n in range(10):
            g.opacities.append([data[n*4:(n+1)*4], data[40+n*4:40+(n+1)*4]])
        return g

    def setOffset(self, offset):
        self.fileOffset = offset

    def getOffset(self):
        return self.fileOffset

class Size:

    def __init__(self):
        self.fileOffset = 0
        self.sizes = []

    @classmethod
    def fromBytes(cls, data):
        g = Size()
        for n in range(10):
            g.sizes.append([data[n*4:(n+1)*4], data[40+n*4:40+(n+1)*4]])
        return g

    def setOffset(self, offset):
        self.fileOffset = offset

    def getOffset(self):
        return self.fileOffset

class ColorGradient:

    def __init__(self):
        self.fileOffset = 0
        self.colors = []

    @classmethod
    def fromBytes(cls, data):
        g = ColorGradient()
        for n in range(10):
            g.colors.append([data[n*4:(n+1)*4], data[40+n*12:40+(n+1)*12]])
        return g

    def setOffset(self, offset):
        self.fileOffset = offset

    def getOffset(self):
        return self.fileOffset

# ==============================================================================
# DATA MODELS - Qt Models for table/tree views
# ==============================================================================

def find_all_occurrences(text, substring):
    indices = []
    start_index = 0
    while True:
        index = text.find(substring, start_index)
        if index == -1:
            break
        indices.append(index)
        start_index = index + 1
    return indices

# ==============================================================================
# DATA MODELS - Qt Models for table/tree views
# ==============================================================================

class SizeModel(QStandardItemModel):
    def __init__(self, undo_stack=None):
        super().__init__()
        self.undo_stack = undo_stack
        self.setHorizontalHeaderLabels(["Time 1", "Size 1", "Time 2", "Size 2", "Time 3", "Size 3", "Time 4", "Size 4", "Time 5", "Size 5", "Time 6", "Size 6", "Time 7", "Size 7", "Time 8", "Size 8", "Time 9", "Size 9", "Time 10", "Size 10"])
        self.sizeGraphs = []

    def setParticleEffect(self, particleEffect):
        self.clear()
        self.sizeGraphs.clear()
        self.particleEffect = particleEffect
        self.setHorizontalHeaderLabels(["Time 1", "Size 1", "Time 2", "Size 2", "Time 3", "Size 3", "Time 4", "Size 4", "Time 5", "Size 5", "Time 6", "Size 6", "Time 7", "Size 7", "Time 8", "Size 8", "Time 9", "Size 9", "Time 10", "Size 10"])
        root = self.invisibleRootItem()
        for particleSystem in self.particleEffect.particle_systems:
            self.sizeGraphs.extend(particleSystem.scale_graphs)
        for graph in self.sizeGraphs:
            if graph is None:
                continue
            arr = []
            for i in range(10):
                timeData = graph.x[i]
                timeItem = QStandardItem(str(timeData))
                if i == 0:
                    timeItem.setData(graph)
                sizeData = graph.y[i]
                sizeItem = QStandardItem(str(sizeData))
                arr.append(timeItem)
                arr.append(sizeItem)
            root.appendRow(arr)

    def setData(self, index, value, role=Qt.EditRole):
        if role == Qt.EditRole and self.undo_stack:
            class Command(QUndoCommand):
                def __init__(self, model, index, value):
                    super().__init__("Edit Size")
                    self.model = model
                    self.index = index
                    self.old = index.data()
                    self.new = value

                def undo(self): self.model._apply(index=self.index, value=self.old)
                def redo(self): self.model._apply(index=self.index, value=self.new)

            self.undo_stack.push(Command(self, index, value))
            return True
        return self._apply(index, value)

    def _apply(self, index, value):
        graph = self.itemFromIndex(index.siblingAtColumn(0)).data()
        i = int(index.column() / 2)
        data = ast.literal_eval(value)
        if index.column() % 2 == 1:
            graph.y[i] = data
        else:
            graph.x[i] = data
        return super().setData(index, value, Qt.EditRole)

class LifetimeModel(QStandardItemModel):

    def __init__(self):
        super().__init__()
        self.setHorizontalHeaderLabels(["Min", "Max"])
        self.lifetime = [0, 0]

    def setParticleEffect(self, particleEffect):
        self.particleEffect = particleEffect
        self.clear()
        self.setHorizontalHeaderLabels(["Min", "Max"])
        root = self.invisibleRootItem()
        minItem = QStandardItem(str(particleEffect.min_lifetime))
        maxItem = QStandardItem(str(particleEffect.max_lifetime))
        root.appendRow([minItem, maxItem])

    def setData(self, index, value, role=Qt.EditRole):
        i = int(index.column()/2)
        data = float(ast.literal_eval(value))
        if index.column() % 2 == 1:
            self.particleEffect.max_lifetime = data
        else:
            self.particleEffect.min_lifetime = data

        return super().setData(index, value, role)

class RotationModel(QStandardItemModel):

    def __init__(self):
        super().__init__()
        self.setHorizontalHeaderLabels(["x axis", "y axis", "z axis"])
        self.rotations = []

    def setFileData(self, fileData):
        self.clear()
        self.rotations.clear()
        self.setHorizontalHeaderLabels(["x axis", "y axis", "z axis"])
        offsets = [x+36 for x in find_all_occurrences(fileData, bytes.fromhex("FFFFFFFFFFFFFFFF00000000FFFFFFFF00000000FFFFFFFF030576F2030576F200000000"))]
        root = self.invisibleRootItem()
        for offset in offsets:
            rotation = EmitterRotation.fromBytes(fileData[offset:offset+48])
            rotation.setOffset(offset)
            self.rotations.append(rotation)
            eulerAngles = rotation.rotation.as_euler('xyz', degrees=True)
            xData, yData, zData = eulerAngles
            xItem = QStandardItem(str(xData))
            xItem.setData(rotation)
            yItem = QStandardItem(str(yData))
            zItem = QStandardItem(str(zData))
            root.appendRow([xItem, yItem, zItem])

    def writeFileData(self, outFile):
        for rotation in self.rotations:
            rotationMatrix = rotation.getRotationMatrix()
            #quaternion = rotation.getQuaternion()
            outFile.seek(rotation.getOffset())
            for index, row in enumerate(rotationMatrix):
                for data in row:
                    outFile.write(struct.pack("<f", data))
                outFile.advance(4)
                #outFile.write(struct.pack("<f", quaternion[index]))

    def setData(self, index, value, role=Qt.EditRole):
        rotation = self.itemFromIndex(index.siblingAtColumn(0)).data()
        data = ast.literal_eval(value)
        euler = rotation.rotation.as_euler('xyz', degrees=True)
        euler[index.column()] = data
        rotation.rotation = Rotation.from_euler('xyz', euler, degrees=True)
        return super().setData(index, value, role)

class PositionModel(QStandardItemModel):

    def __init__(self):
        super().__init__()
        self.setHorizontalHeaderLabels(["x offset", "y offset", "z offset"])
        self.positions = []

    def setFileData(self, fileData):
        self.clear()
        self.positions.clear()
        self.setHorizontalHeaderLabels(["x offset", "y offset", "z offset"])
        offsets = [x+84 for x in find_all_occurrences(fileData, bytes.fromhex("FFFFFFFFFFFFFFFF00000000FFFFFFFF00000000FFFFFFFF030576F2030576F200000000"))]
        root = self.invisibleRootItem()
        for offset in offsets:
            position = EmitterPosition.fromBytes(fileData[offset:offset+12])
            position.setOffset(offset)
            self.positions.append(position)
            xData = struct.unpack("<f", position.position[0])[0]
            yData = struct.unpack("<f", position.position[1])[0]
            zData = struct.unpack("<f", position.position[2])[0]
            xItem = QStandardItem(str(xData))
            xItem.setData(position)
            yItem = QStandardItem(str(yData))
            zItem = QStandardItem(str(zData))
            root.appendRow([xItem, yItem, zItem])

    def writeFileData(self, outFile):
        for position in self.positions:
            outFile.seek(position.getOffset())
            outFile.write(position.position[0])
            outFile.write(position.position[1])
            outFile.write(position.position[2])

    def setData(self, index, value, role=Qt.EditRole):
        position = self.itemFromIndex(index.siblingAtColumn(0)).data()
        data = ast.literal_eval(value)
        position.position[index.column()] = struct.pack("<f", data)
        return super().setData(index, value, role)

class OpacityGradientModel(QStandardItemModel):
    def __init__(self, undo_stack=None):
        super().__init__()
        self.undo_stack = undo_stack
        self.file = MemoryStream()
        self.setHorizontalHeaderLabels(["Time 1", "Opacity 1", "Time 2", "Opacity 2", "Time 3", "Opacity 3", "Time 4", "Opacity 4", "Time 5", "Opacity 5", "Time 6", "Opacity 6", "Time 7", "Opacity 7", "Time 8", "Opacity 8", "Time 9", "Opacity 9", "Time 10", "Opacity 10"])
        self.opacityGraphs = []

    def setParticleEffect(self, particleEffect):
        self.clear()
        self.opacityGraphs.clear()
        self.particleEffect = particleEffect
        self.setHorizontalHeaderLabels(["Time 1", "Opacity 1", "Time 2", "Opacity 2", "Time 3", "Opacity 3", "Time 4", "Opacity 4", "Time 5", "Opacity 5", "Time 6", "Opacity 6", "Time 7", "Opacity 7", "Time 8", "Opacity 8", "Time 9", "Opacity 9", "Time 10", "Opacity 10"])
        for particleSystem in self.particleEffect.particle_systems:
            self.opacityGraphs.extend(particleSystem.opacity_graphs)
        root = self.invisibleRootItem()
        for graph in self.opacityGraphs:
            arr = []
            for i in range(10):
                timeData = graph.x[i]
                timeItem = QStandardItem(str(timeData))
                if i == 0:
                    timeItem.setData(graph)
                opacityData = graph.y[i]
                opacityItem = QStandardItem(str(opacityData))
                arr.append(timeItem)
                arr.append(opacityItem)
            root.appendRow(arr)

    def setData(self, index, value, role=Qt.EditRole):
        if role == Qt.EditRole and self.undo_stack:
            class Command(QUndoCommand):
                def __init__(self, model, index, value):
                    super().__init__("Edit Opacity")
                    self.model = model
                    self.index = index
                    self.old = index.data()
                    self.new = value

                def undo(self): self.model._apply(index=self.index, value=self.old)
                def redo(self): self.model._apply(index=self.index, value=self.new)

            self.undo_stack.push(Command(self, index, value))
            return True
        return self._apply(index, value)

    def _apply(self, index, value):
        graph = self.itemFromIndex(index.siblingAtColumn(0)).data()
        i = int(index.column() / 2)
        data = ast.literal_eval(value)
        if index.column() % 2 == 1:
            graph.y[i] = data
        else:
            graph.x[i] = data
        return super().setData(index, value, Qt.EditRole)

class ColorGradientModel(QStandardItemModel):
    def __init__(self, undo_stack=None):
        super().__init__()
        self.undo_stack = undo_stack
        self.setHorizontalHeaderLabels(["Time 1", "Color 1", "Time 2", "Color 2", "Time 3", "Color 3", "Time 4", "Color 4", "Time 5", "Color 5", "Time 6", "Color 6", "Time 7", "Color 7", "Time 8", "Color 8", "Time 9", "Color 9", "Time 10", "Color 10"])
        self.colorGraphs = []

    def setParticleEffect(self, particleEffect):
        self.particleEffect = particleEffect
        self.clear()
        self.colorGraphs.clear()
        self.setHorizontalHeaderLabels(["Time 1", "Color 1", "Time 2", "Color 2", "Time 3", "Color 3", "Time 4", "Color 4", "Time 5", "Color 5", "Time 6", "Color 6", "Time 7", "Color 7", "Time 8", "Color 8", "Time 9", "Color 9", "Time 10", "Color 10"])
        for particleSystem in self.particleEffect.particle_systems:
            self.colorGraphs.extend(particleSystem.color_graphs)
        root = self.invisibleRootItem()
        for graph in self.colorGraphs:
            arr = []
            for i in range(10):
                timeData = graph.x[i]
                timeItem = QStandardItem(str(timeData))
                if i == 0:
                    timeItem.setData(graph)
                colorData = graph.y[i]
                colorItem = QStandardItem(str(colorData))
                arr.append(timeItem)
                arr.append(colorItem)
            root.appendRow(arr)

    def setData(self, index, value, role=Qt.EditRole):
        if role == Qt.EditRole and self.undo_stack:
            class Command(QUndoCommand):
                def __init__(self, model, index, value):
                    super().__init__("Edit Color")
                    self.model = model
                    self.index = index
                    self.old = index.data()
                    self.new = value

                def undo(self): self.model._apply(index=self.index, value=self.old)
                def redo(self): self.model._apply(index=self.index, value=self.new)

            self.undo_stack.push(Command(self, index, value))
            return True
        return self._apply(index, value)

    def _apply(self, index, value):
        graph = self.itemFromIndex(index.siblingAtColumn(0)).data()
        i = int(index.column() / 2)
        data = ast.literal_eval(value)
        if index.column() % 2 == 1:
            graph.y[i] = data
        else:
            graph.x[i] = data
        return super().setData(index, value, Qt.EditRole)

class OpacityTable(QTableView):

    def __init__(self, parent=None):
        super().__init__(parent)
        paste_shortcut = QShortcut(QKeySequence("Ctrl+V"), self)
        paste_shortcut.activated.connect(self.pasteFromClipboard)

        
    def pasteFromClipboard(self):
        clipboard = QApplication.clipboard()
        text = clipboard.text().strip()
        if not text:
            return

        selected = self.selectedIndexes()
        if not selected:
            return

        model: QAbstractItemModel = self.model()

        rows = text.split('\n')
        if len(rows) == 1 and '\t' not in text:
            # Single value: apply to all selected cells
            for index in selected:
                if index.isValid():
                    model.setData(index, text)
        else:
            # Multi-value paste starting from top-left
            data = [row.split('\t') for row in rows]
            top_left = sorted(selected, key=lambda idx: (idx.row(), idx.column()))[0]
            start_row = top_left.row()
            start_col = top_left.column()

            for r, row_data in enumerate(data):
                for c, cell in enumerate(row_data):
                    model_index = model.index(start_row + r, start_col + c)
                    if model_index.isValid():
                        model.setData(model_index, cell)

class ColorTable(QTableView):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mainWindow = parent
        self.initUI()

    def initUI(self):
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.showContextMenu)
        self.contextMenuColorPickerAction = QAction("Color Picker")
        self.contextMenuColorPickerAction.triggered.connect(self.showColorPicker)
        self.contextMenuHuePickerAction = QAction("Hue Picker")
        self.contextMenuHuePickerAction.triggered.connect(self.showHuePicker)
        self.contextMenuMultiColorPickerAction = QAction("Color Picker")
        self.contextMenuMultiColorPickerAction.triggered.connect(self.showMultiColorPicker)
        self.contextMenu = QMenu(self)

        # Add Ctrl+V shortcut
        paste_shortcut = QShortcut(QKeySequence("Ctrl+V"), self)
        paste_shortcut.activated.connect(self.pasteFromClipboard)

    def _getColorGraphsForEffect(self, effect):
        """Build the flat color_graphs list for an effect (same order as ColorGradientModel)."""
        graphs = []
        if effect and hasattr(effect, 'particle_systems'):
            for ps in effect.particle_systems:
                if hasattr(ps, 'color_graphs'):
                    graphs.extend(ps.color_graphs)
        return graphs

    def _applyColorToOtherFiles(self, applyFunc):
        """Apply a color transform function to all other files' selected color cells.
        applyFunc(currentRgbTuple) -> newRgbTuple or None to skip.
        """
        if not self.mainWindow:
            return
        self.mainWindow.saveCurrentSelectionState()
        loadedFiles = self.mainWindow.loadedFilesStrip.getAllLoadedFiles()
        currentPath = self.mainWindow.currentFilePath

        for filepath, fileData, effect, note in loadedFiles:
            if filepath == currentPath:
                continue
            if filepath not in self.mainWindow.fileSelectionStates:
                continue
            selection = self.mainWindow.fileSelectionStates[filepath]['selection']
            if not selection:
                continue

            colorGraphs = self._getColorGraphsForEffect(effect)
            for row, col in selection:
                if col % 2 != 1:  # Skip non-color columns (time columns)
                    continue
                if row >= len(colorGraphs):
                    continue
                graph = colorGraphs[row]
                colorIdx = col // 2
                if colorIdx >= len(graph.y):
                    continue
                try:
                    oldRgb = tuple(int(c) for c in graph.y[colorIdx])
                    newRgb = applyFunc(oldRgb)
                    if newRgb is not None:
                        # Validate and clamp RGB values to [0, 255]
                        validated = []
                        for val in newRgb:
                            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                                print(f"Invalid color value detected: {val}, skipping")
                                validated = None
                                break
                            validated.append(max(0.0, min(255.0, float(val))))
                        
                        if validated is not None:
                            graph.y[colorIdx] = validated
                except Exception as e:
                    print(f"Error applying color at row {row}, col {col}: {e}")
                    pass

    def showColorPicker(self, pos):
        assert(len(self.selectedIndexes()) == 1)
        index = self.selectedIndexes()[0]
        colorTuple = ast.literal_eval(self.model().itemFromIndex(index).text())
        color = QColor(*colorTuple)
        selectedColor = QColorDialog.getColor(initial=color, parent=self, title="Select New Color")
        if not selectedColor.isValid():
            return
        try:
            colorTuple = selectedColor.toTuple()[0:3]
            self.model().setData(index, str(colorTuple))
        except:
            pass

        # Apply same color to other files' selected cells
        newH, newS, newV = selectedColor.hue(), selectedColor.saturation(), selectedColor.value()
        def applyExactColor(oldRgb):
            c = QColor(*oldRgb)
            c.setHsv(newH, newS, newV)
            return c.toRgb().toTuple()[0:3]
        self._applyColorToOtherFiles(applyExactColor)
            
    def showMultiColorPicker(self, pos):
        index = [i for i in self.selectedIndexes() if i.column() % 2 == 1][0]
        colorTuple = ast.literal_eval(self.model().itemFromIndex(index).text())
        color = QColor(*colorTuple)
        selectedColor = QColorDialog.getColor(initial=color, parent=self, title="Select New Color")
        if not selectedColor.isValid():
            return
        colors = [(QColor(*ast.literal_eval(self.model().itemFromIndex(i).text())), i) for i in self.selectedIndexes() if i.column() % 2 == 1]
        for color, index in colors:
            try:
                color.setHsv(selectedColor.hue(), selectedColor.saturation(), selectedColor.value())
                colorTuple = color.toRgb().toTuple()[0:3]
                self.model().setData(index, str(colorTuple))
            except:
                pass

        # Apply same color to other files' selected cells
        newH, newS, newV = selectedColor.hue(), selectedColor.saturation(), selectedColor.value()
        def applyColor(oldRgb):
            c = QColor(*oldRgb)
            c.setHsv(newH, newS, newV)
            return c.toRgb().toTuple()[0:3]
        self._applyColorToOtherFiles(applyColor)
            
    def showHuePicker(self, pos):
        # Save current file state
        if self.mainWindow:
            self.mainWindow.saveCurrentSelectionState()
        
        index = [i for i in self.selectedIndexes() if i.column() % 2 == 1][0]
        colorTuple = ast.literal_eval(self.model().itemFromIndex(index).text())
        color = QColor(*colorTuple)
        selectedColor = QColorDialog.getColor(initial=color, parent=self, title="Adjust color hue")
        if not selectedColor.isValid():
            return
        hue = selectedColor.hue()
        
        # Apply hue change to current file
        colors = [(QColor(*ast.literal_eval(self.model().itemFromIndex(i).text())), i) for i in self.selectedIndexes() if i.column() % 2 == 1]
        for color, index in colors:
            try:
                color.setHsv(hue, color.saturation(), color.value())
                colorTuple = color.toRgb().toTuple()[0:3]
                self.model().setData(index, str(colorTuple))
            except:
                pass
        
        # Apply hue change to ALL other files with selections
        def applyHue(oldRgb):
            c = QColor(*oldRgb)
            c.setHsv(hue, c.saturation(), c.value())
            return c.toRgb().toTuple()[0:3]
        self._applyColorToOtherFiles(applyHue)

    def triggerColorPickerFromButton(self):
        if not self.selectedIndexes():
            return
        validIndexes = [i for i in self.selectedIndexes() if i.column() % 2 == 1]
        selected = self.selectedIndexes()
        if len(validIndexes) > 1:
            self.showMultiColorPicker(None)
        elif len(validIndexes) == 1:
            self.showColorPicker(None)  # We ignore 'pos' in showColorPicker anyway


    def showContextMenu(self, pos):
        self.contextMenu.clear()
        if not self.selectedIndexes():
            return
        validIndexes = [i for i in self.selectedIndexes() if i.column() % 2 == 1]
        if len(validIndexes) > 1:
            self.contextMenu.addAction(self.contextMenuHuePickerAction)
            self.contextMenu.addAction(self.contextMenuMultiColorPickerAction)
            global_pos = self.mapToGlobal(pos)
            self.contextMenu.exec(global_pos)
        elif len(validIndexes) == 1:
            self.contextMenu.addAction(self.contextMenuColorPickerAction)
            global_pos = self.mapToGlobal(pos)
            self.contextMenu.exec(global_pos)

    def pasteFromClipboard(self):
        clipboard = QApplication.clipboard()
        text = clipboard.text().strip()
        if not text:
            return

        selected = self.selectedIndexes()
        if not selected:
            return

        model: QAbstractItemModel = self.model()

        rows = text.split('\n')
        if len(rows) == 1 and '\t' not in text:
            # Single value: apply to all selected cells
            for index in selected:
                if index.isValid():
                    model.setData(index, text)
        else:
            # Multi-value paste starting from top-left
            data = [row.split('\t') for row in rows]
            top_left = sorted(selected, key=lambda idx: (idx.row(), idx.column()))[0]
            start_row = top_left.row()
            start_col = top_left.column()

            for r, row_data in enumerate(data):
                for c, cell in enumerate(row_data):
                    model_index = model.index(start_row + r, start_col + c)
                    if model_index.isValid():
                        model.setData(model_index, cell)

class ColorSwatchDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        text = index.data()
        if not text:
            super().paint(painter, option, index)
            return

        # Clean and parse RGB
        cleaned_text = text.strip().lstrip("(").rstrip(")").lstrip("[").rstrip("]")

        try:
            parts = [float(x.strip()) for x in cleaned_text.split(",")]
            if len(parts) != 3:
                raise ValueError("Not 3 components")
            r, g, b = [max(0, min(255, int(c))) for c in parts]
            color = QColor(r, g, b)
        except Exception:
            super().paint(painter, option, index)
            return

        # Draw selection background if selected
        if option.state & QStyle.State_Selected:
            painter.fillRect(option.rect, option.palette.highlight())

        # Draw color swatch
        swatch_size = 16
        swatch_rect = QRect(
            option.rect.left() + 4,
            option.rect.center().y() - swatch_size // 2,
            swatch_size,
            swatch_size
        )
        painter.setPen(Qt.black)
        painter.setBrush(color)
        painter.drawRect(swatch_rect)

        # Draw the RGB text next to the swatch
        text_rect = QRect(
            swatch_rect.right() + 6,
            option.rect.top(),
            option.rect.width() - swatch_size - 10,
            option.rect.height()
        )
        painter.setPen(
            option.palette.highlightedText().color()
            if option.state & QStyle.State_Selected
            else Qt.black
        )
        painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, text)

# ==============================================================================
# FILE MANAGEMENT COMPONENTS - Widgets for managing loaded particle files
# ==============================================================================
        
class LoadedFilesWindow(QWidget):
    
    loadFile = Signal(str, MemoryStream, ParticleEffect)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        
        self.treeWidget = QTreeWidget(self)
        self.treeWidget.setHeaderLabels(["Loaded Files"])
        
        self.layout.addWidget(self.treeWidget)
        
        self.setLayout(self.layout)
        
    def addFile(self, filepath, fileData, effect, note=""):
        fileWidget = LoadedFileWidget(filepath, fileData, effect, note=note)
        fileWidget.openClicked.connect(self.load)
        fileWidget.removeClicked.connect(self.remove)
        item = QTreeWidgetItem(self.treeWidget)
        fileWidget.item = item
        self.treeWidget.setItemWidget(item, 0, fileWidget)
        
    def getAllLoadedFiles(self):
        fileWidgets = []
        root = self.treeWidget.invisibleRootItem()
        for i in range(root.childCount()):
            fileWidgets.append(self.treeWidget.itemWidget(root.child(i), 0))
        return fileWidgets
        
    def remove(self, item):
        (item.parent() or self.treeWidget.invisibleRootItem()).removeChild(item)
        
    def load(self, filepath: str, fileData: MemoryStream, particleEffect: ParticleEffect):
        self.loadFile.emit(filepath, fileData, particleEffect)
        
    def clear(self):
        self.treeWidget.clear()
        
        
class LoadedFileWidget(QWidget):
    
    openClicked = Signal(str, MemoryStream, ParticleEffect)
    removeClicked = Signal(QTreeWidgetItem)
    
    def __init__(self, filepath, fileData, effect, note="", parent=None):
        super().__init__(parent)
        self.filepath = filepath
        self.fileData = fileData
        self.particleEffect = effect
        self.note = note
        
        self.layout = QHBoxLayout()
        
        self.selectButton = QToolButton()
        self.selectButton.setText("open")
        self.selectButton.clicked.connect(self.load)
        self.nameLabel = QLabel(os.path.basename(self.filepath), parent=self)
        self.noteEdit = QLineEdit(parent=self)
        self.noteEdit.setText(self.note)
        self.noteEdit.editingFinished.connect(self.setNote)
        self.noteEdit.setPlaceholderText("Set note...")
        self.removeButton = QToolButton()
        self.removeButton.setText("\u2715")
        self.removeButton.clicked.connect(self.remove)
        
        self.layout.addWidget(self.selectButton)
        self.layout.addWidget(self.nameLabel)
        self.layout.addWidget(self.noteEdit)
        #self.layout.addStretch(1)
        self.layout.addWidget(self.removeButton)
        self.setLayout(self.layout)
        
    def setNote(self):
        self.note = self.noteEdit.text()
        
    def remove(self):
        self.removeClicked.emit(self.item)
        
    def load(self):
        self.openClicked.emit(self.filepath, self.fileData, self.particleEffect)

class ParticleFileListItem(QWidget):
    """Custom widget for each file in the list with color preview"""
    
    clearSelectionClicked = Signal(str)  # Signal when X button clicked
    
    def __init__(self, filepath, particleEffect, parent=None):
        super().__init__(parent)
        self.filepath = filepath
        self.particleEffect = particleEffect
        
        # Main horizontal layout
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 3, 5, 3)
        layout.setSpacing(8)
        
        # Selection indicator (tick mark)
        self.tickLabel = QLabel("✓")
        self.tickLabel.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 14px;")
        self.tickLabel.setVisible(False)
        layout.addWidget(self.tickLabel)
        
        # File name label
        self.nameLabel = QLabel(os.path.basename(filepath))
        self.nameLabel.setMinimumWidth(180)
        layout.addWidget(self.nameLabel)
        
        # Clear selection button (X)
        self.clearBtn = QToolButton()
        self.clearBtn.setText("✕")
        self.clearBtn.setStyleSheet("color: #f44336; font-weight: bold;")
        self.clearBtn.setFixedSize(20, 20)
        self.clearBtn.setToolTip("Clear all selections in this file")
        self.clearBtn.clicked.connect(lambda: self.clearSelectionClicked.emit(self.filepath))
        self.clearBtn.setVisible(False)
        layout.addWidget(self.clearBtn)
        
        # Spacer
        layout.addStretch()
        
        # Color preview container
        colorContainer = QWidget()
        colorLayout = QHBoxLayout()
        colorLayout.setContentsMargins(0, 0, 0, 0)
        colorLayout.setSpacing(3)
        
        # Extract unique colors from particle effect
        uniqueColors = self.extractUniqueColors()
        
        # Create color squares (limit to 20 to avoid overcrowding)
        for color in uniqueColors[:20]:
            colorSquare = QFrame()
            colorSquare.setFixedSize(20, 20)
            # Ensure values are in 0-1 range then convert to 0-255
            r, g, b = [int(max(0, min(1, c)) * 255) for c in color]
            colorSquare.setStyleSheet(f"background-color: rgb({r}, {g}, {b}); border: 1px solid #666;")
            colorSquare.setToolTip(f"RGB: ({r}, {g}, {b})")
            colorLayout.addWidget(colorSquare)
        
        if len(uniqueColors) > 20:
            moreLabel = QLabel(f"+{len(uniqueColors) - 20}")
            moreLabel.setStyleSheet("color: #888; font-size: 10px;")
            colorLayout.addWidget(moreLabel)
        
        colorContainer.setLayout(colorLayout)
        layout.addWidget(colorContainer)
        
        self.setLayout(layout)
    
    def extractUniqueColors(self):
        """Extract unique colors from particle effect"""
        colors = []
        all_colors = []  # Keep track of all colors including dark ones
        
        if not self.particleEffect or not hasattr(self.particleEffect, 'particle_systems'):
            return colors
        
        # Collect all colors from all particle systems
        for system in self.particleEffect.particle_systems:
            if hasattr(system, 'color_graphs'):
                for colorGraph in system.color_graphs:
                    if hasattr(colorGraph, 'y'):
                        for color in colorGraph.y:
                            # Normalize color values - check if already in 0-255 range or 0-1 range
                            normalized_color = []
                            for c in color:
                                # If value > 1, assume it's in 0-255 range, convert to 0-1
                                if c > 1.0:
                                    normalized_color.append(c / 255.0)
                                else:
                                    # Already in 0-1 range, clamp to prevent overflow
                                    normalized_color.append(max(0.0, min(1.0, c)))
                            
                            # Convert to tuple for comparison
                            colorTuple = tuple(round(c, 3) for c in normalized_color)
                            
                            # Check if color already exists in all_colors list
                            exists = False
                            for existing in all_colors:
                                if tuple(round(c, 3) for c in existing) == colorTuple:
                                    exists = True
                                    break
                            
                            if not exists:
                                all_colors.append(normalized_color)
                                # Skip very dark colors for the main list
                                if sum(normalized_color) > 0.05:  # Changed from all() check to sum check
                                    colors.append(normalized_color)
        
        # If no colors found after filtering, return all colors (including dark ones)
        # This ensures we always show something
        if len(colors) == 0 and len(all_colors) > 0:
            return all_colors
        
        return colors
    
    def setEdited(self, value: bool):
        """Mark the file as edited"""
        basename = os.path.basename(self.filepath)
        self.nameLabel.setText(f"{basename}{'*' if value else ''}")
    
    def setHasSelection(self, hasSelection: bool):
        """Show/hide selection indicators"""
        self.tickLabel.setVisible(hasSelection)
        self.clearBtn.setVisible(hasSelection)

class LoadedFilesBar(QWidget):
    
    loadFile = Signal(str, MemoryStream, ParticleEffect)
    
    GROUP_ROLE = Qt.ItemDataRole.UserRole + 1      # True if group item
    FILEPATH_ROLE = Qt.ItemDataRole.UserRole + 2   # filepath string for file items
    COLOR_ROLE = Qt.ItemDataRole.UserRole + 3      # group color string
    DATA_INDEX_ROLE = Qt.ItemDataRole.UserRole + 4 # index into fileData list
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.mainWindow = parent
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Search box
        self.searchBox = QLineEdit()
        self.searchBox.setPlaceholderText("Search by filename or ID...")
        self.searchBox.textChanged.connect(self.filterFiles)
        
        # Tree widget for files (supports grouping)
        self.treeWidget = QTreeWidget(self)
        self.treeWidget.setHeaderHidden(True)
        self.treeWidget.setSelectionMode(QTreeWidget.ExtendedSelection)
        self.treeWidget.setDragDropMode(QTreeWidget.InternalMove)
        self.treeWidget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.treeWidget.customContextMenuRequested.connect(self.showContextMenu)
        self.treeWidget.currentItemChanged.connect(self.onCurrentItemChanged)
        
        # Note entry
        self.noteEntry = QLineEdit()
        self.noteEntry.setPlaceholderText("Set Note:")
        self.noteEntry.textEdited.connect(self.setNote)
        
        self.layout.addWidget(self.searchBox)
        self.layout.addWidget(self.treeWidget)
        self.layout.addWidget(self.noteEntry)
        
        self.fileData = []  # Store [filepath, fileData, effect, note]
        self._groupCounter = 0
        
        self.setLayout(self.layout)
    
    def _getFileDataIndex(self, filepath):
        """Get index in fileData by filepath"""
        for i, data in enumerate(self.fileData):
            if data[0] == filepath:
                return i
        return -1
    
    def onCurrentItemChanged(self, current, previous):
        """Handle tree item selection change"""
        if not current:
            return
        
        isGroup = current.data(0, self.GROUP_ROLE)
        if isGroup:
            return
        
        filepath = current.data(0, self.FILEPATH_ROLE)
        if not filepath:
            return
        
        idx = self._getFileDataIndex(filepath)
        if idx == -1:
            return
        
        data = self.fileData[idx]
        if data[3]:
            self.noteEntry.setText(data[3])
        else:
            self.noteEntry.setText("")
        self.loadFile.emit(data[0], data[1], data[2])
        
    def addFile(self, filepath, fileData, effect, note=""):
        self.fileData.append([filepath, fileData, effect, note])
        
        # Create tree item
        item = QTreeWidgetItem()
        item.setData(0, self.GROUP_ROLE, False)
        item.setData(0, self.FILEPATH_ROLE, filepath)
        
        # Create custom widget
        itemWidget = ParticleFileListItem(filepath, effect)
        itemWidget.clearSelectionClicked.connect(self.clearFileSelection)
        
        self.treeWidget.addTopLevelItem(item)
        self.treeWidget.setItemWidget(item, 0, itemWidget)
        item.setSizeHint(0, itemWidget.sizeHint())
        
        self.treeWidget.setCurrentItem(item)
        
    def closeCurrentTab(self):
        current = self.treeWidget.currentItem()
        if not current:
            return
        isGroup = current.data(0, self.GROUP_ROLE)
        if isGroup:
            return
        filepath = current.data(0, self.FILEPATH_ROLE)
        if filepath is None:
            return
        idx = self._getFileDataIndex(filepath)
        if idx >= 0:
            del self.fileData[idx]
        # Remove from tree
        parent = current.parent()
        if parent:
            parent.removeChild(current)
        else:
            index = self.treeWidget.indexOfTopLevelItem(current)
            if index >= 0:
                self.treeWidget.takeTopLevelItem(index)
            
    def getAllLoadedFiles(self):
        return self.fileData
        
    def getSelectedFile(self):
        current = self.treeWidget.currentItem()
        if current and not current.data(0, self.GROUP_ROLE):
            filepath = current.data(0, self.FILEPATH_ROLE)
            idx = self._getFileDataIndex(filepath)
            if idx >= 0:
                return self.fileData[idx]
        return None
        
    def setCurrentFilePath(self, newPath):
        current = self.treeWidget.currentItem()
        if current and not current.data(0, self.GROUP_ROLE):
            oldPath = current.data(0, self.FILEPATH_ROLE)
            idx = self._getFileDataIndex(oldPath)
            if idx >= 0:
                self.fileData[idx][0] = newPath
                current.setData(0, self.FILEPATH_ROLE, newPath)
                widget = self.treeWidget.itemWidget(current, 0)
                if widget:
                    widget.filepath = newPath
                    widget.nameLabel.setText(os.path.basename(newPath))
        
    def setNote(self, newNote):
        current = self.treeWidget.currentItem()
        if current and not current.data(0, self.GROUP_ROLE):
            filepath = current.data(0, self.FILEPATH_ROLE)
            idx = self._getFileDataIndex(filepath)
            if idx >= 0:
                self.fileData[idx][3] = newNote
            
    def markEdited(self, value: bool):
        current = self.treeWidget.currentItem()
        if current and not current.data(0, self.GROUP_ROLE):
            widget = self.treeWidget.itemWidget(current, 0)
            if widget:
                widget.setEdited(value)
    
    def filterFiles(self, searchText):
        """Filter files based on search text"""
        searchLower = searchText.lower()
        
        def filterItem(item):
            isGroup = item.data(0, self.GROUP_ROLE)
            if isGroup:
                # Show group if any child matches
                anyVisible = False
                for i in range(item.childCount()):
                    child = item.child(i)
                    childVisible = filterItem(child)
                    if childVisible:
                        anyVisible = True
                item.setHidden(not anyVisible and bool(searchText))
                # Expand groups with matches
                if anyVisible:
                    item.setExpanded(True)
                return anyVisible
            else:
                widget = self.treeWidget.itemWidget(item, 0)
                if widget:
                    filename = os.path.basename(widget.filepath).lower()
                    visible = searchLower in filename
                    item.setHidden(not visible)
                    return visible
                return True
        
        for i in range(self.treeWidget.topLevelItemCount()):
            filterItem(self.treeWidget.topLevelItem(i))
    
    def showContextMenu(self, pos):
        """Show context menu for file tree"""
        selectedItems = self.treeWidget.selectedItems()
        
        menu = QMenu(self)
        
        # Get file items only (non-group)
        fileItems = [item for item in selectedItems if not item.data(0, self.GROUP_ROLE)]
        groupItems = [item for item in selectedItems if item.data(0, self.GROUP_ROLE)]
        
        if len(fileItems) >= 2:
            groupAction = QAction(f"Group {len(fileItems)} files", self)
            groupAction.triggered.connect(lambda: self.groupSelectedItems(fileItems))
            menu.addAction(groupAction)
            menu.addSeparator()
        
        if len(groupItems) == 1:
            group = groupItems[0]
            renameAction = QAction("Rename Group", self)
            renameAction.triggered.connect(lambda: self.renameGroup(group))
            menu.addAction(renameAction)
            
            colorAction = QAction("Change Group Color", self)
            colorAction.triggered.connect(lambda: self.changeGroupColor(group))
            menu.addAction(colorAction)
            
            ungroupAction = QAction("Ungroup", self)
            ungroupAction.triggered.connect(lambda: self.ungroupItems(group))
            menu.addAction(ungroupAction)
            menu.addSeparator()
        
        if len(fileItems) >= 1:
            deleteAction = QAction(f"Delete ({len(fileItems)} file{'s' if len(fileItems) > 1 else ''})", self)
            deleteAction.triggered.connect(lambda: self.deleteSelectedFiles(fileItems))
            menu.addAction(deleteAction)
        
        if len(groupItems) == 1 and len(fileItems) == 0:
            deleteGroupAction = QAction("Delete Group & Files", self)
            deleteGroupAction.triggered.connect(lambda: self.deleteGroup(groupItems[0]))
            menu.addAction(deleteGroupAction)
        
        if menu.actions():
            menu.exec(self.treeWidget.mapToGlobal(pos))
    
    def groupSelectedItems(self, fileItems):
        """Group selected file items into a folder"""
        self._groupCounter += 1
        groupName = f"Group {self._groupCounter}"
        groupColor = "#4a90d9"
        
        # Create group item
        groupItem = QTreeWidgetItem()
        groupItem.setData(0, self.GROUP_ROLE, True)
        groupItem.setData(0, self.COLOR_ROLE, groupColor)
        groupItem.setText(0, f"📁 {groupName}")
        groupItem.setForeground(0, QColor(groupColor))
        groupItem.setFlags(groupItem.flags() | Qt.ItemFlag.ItemIsEditable)
        
        # Find insertion index (where the first selected item is)
        firstItem = fileItems[0]
        firstParent = firstItem.parent()
        if firstParent:
            insertIdx = firstParent.indexOfChild(firstItem)
            firstParent.insertChild(insertIdx, groupItem)
        else:
            insertIdx = self.treeWidget.indexOfTopLevelItem(firstItem)
            self.treeWidget.insertTopLevelItem(insertIdx, groupItem)
        
        # Move items into group
        for item in fileItems:
            # Save widget before removing
            widget = self.treeWidget.itemWidget(item, 0)
            filepath = item.data(0, self.FILEPATH_ROLE)
            
            # Remove from current parent
            parent = item.parent()
            if parent:
                idx = parent.indexOfChild(item)
                parent.takeChild(idx)
            else:
                idx = self.treeWidget.indexOfTopLevelItem(item)
                self.treeWidget.takeTopLevelItem(idx)
            
            # Add to group
            groupItem.addChild(item)
            
            # Re-create widget (widget is lost on reparent)
            if filepath:
                dataIdx = self._getFileDataIndex(filepath)
                if dataIdx >= 0:
                    effect = self.fileData[dataIdx][2]
                    newWidget = ParticleFileListItem(filepath, effect)
                    newWidget.clearSelectionClicked.connect(self.clearFileSelection)
                    self.treeWidget.setItemWidget(item, 0, newWidget)
                    item.setSizeHint(0, newWidget.sizeHint())
        
        groupItem.setExpanded(True)
    
    def ungroupItems(self, groupItem):
        """Ungroup: move all children to top level"""
        parentOfGroup = groupItem.parent()
        if parentOfGroup:
            insertIdx = parentOfGroup.indexOfChild(groupItem)
        else:
            insertIdx = self.treeWidget.indexOfTopLevelItem(groupItem)
        
        children = []
        while groupItem.childCount() > 0:
            child = groupItem.takeChild(0)
            children.append(child)
        
        # Insert children at group's position
        for i, child in enumerate(children):
            filepath = child.data(0, self.FILEPATH_ROLE)
            if parentOfGroup:
                parentOfGroup.insertChild(insertIdx + i, child)
            else:
                self.treeWidget.insertTopLevelItem(insertIdx + i, child)
            # Re-create widget
            if filepath:
                dataIdx = self._getFileDataIndex(filepath)
                if dataIdx >= 0:
                    effect = self.fileData[dataIdx][2]
                    newWidget = ParticleFileListItem(filepath, effect)
                    newWidget.clearSelectionClicked.connect(self.clearFileSelection)
                    self.treeWidget.setItemWidget(child, 0, newWidget)
                    child.setSizeHint(0, newWidget.sizeHint())
        
        # Remove empty group
        if parentOfGroup:
            parentOfGroup.removeChild(groupItem)
        else:
            idx = self.treeWidget.indexOfTopLevelItem(groupItem)
            self.treeWidget.takeTopLevelItem(idx)
    
    def renameGroup(self, groupItem):
        """Rename a group via dialog"""
        from PySide6.QtWidgets import QInputDialog
        currentName = groupItem.text(0).replace("📁 ", "")
        newName, ok = QInputDialog.getText(self, "Rename Group", "Group name:", text=currentName)
        if ok and newName:
            color = groupItem.data(0, self.COLOR_ROLE) or "#4a90d9"
            groupItem.setText(0, f"📁 {newName}")
            groupItem.setForeground(0, QColor(color))
    
    def changeGroupColor(self, groupItem):
        """Change group color via color dialog"""
        currentColor = groupItem.data(0, self.COLOR_ROLE) or "#4a90d9"
        color = QColorDialog.getColor(QColor(currentColor), self, "Select Group Color")
        if color.isValid():
            colorStr = color.name()
            groupItem.setData(0, self.COLOR_ROLE, colorStr)
            groupItem.setForeground(0, QColor(colorStr))
            currentText = groupItem.text(0)
            groupItem.setText(0, currentText)  # Refresh display
    
    def deleteSelectedFiles(self, fileItems):
        """Delete selected file items"""
        for item in fileItems:
            filepath = item.data(0, self.FILEPATH_ROLE)
            if filepath:
                idx = self._getFileDataIndex(filepath)
                if idx >= 0:
                    del self.fileData[idx]
                # Remove selection state
                if self.mainWindow and filepath in self.mainWindow.fileSelectionStates:
                    del self.mainWindow.fileSelectionStates[filepath]
            # Remove from tree
            parent = item.parent()
            if parent:
                parent.removeChild(item)
            else:
                idx = self.treeWidget.indexOfTopLevelItem(item)
                if idx >= 0:
                    self.treeWidget.takeTopLevelItem(idx)
    
    def deleteGroup(self, groupItem):
        """Delete a group and all its file children"""
        filesToDelete = []
        for i in range(groupItem.childCount()):
            child = groupItem.child(i)
            if not child.data(0, self.GROUP_ROLE):
                filesToDelete.append(child)
        
        for item in filesToDelete:
            filepath = item.data(0, self.FILEPATH_ROLE)
            if filepath:
                idx = self._getFileDataIndex(filepath)
                if idx >= 0:
                    del self.fileData[idx]
                if self.mainWindow and filepath in self.mainWindow.fileSelectionStates:
                    del self.mainWindow.fileSelectionStates[filepath]
        
        parent = groupItem.parent()
        if parent:
            parent.removeChild(groupItem)
        else:
            idx = self.treeWidget.indexOfTopLevelItem(groupItem)
            if idx >= 0:
                self.treeWidget.takeTopLevelItem(idx)
    
    def updateFileIndicators(self):
        """Update tick/X indicators for all files based on selection state"""
        if not self.mainWindow:
            return
        
        def updateItem(item):
            isGroup = item.data(0, self.GROUP_ROLE)
            if isGroup:
                for i in range(item.childCount()):
                    updateItem(item.child(i))
            else:
                widget = self.treeWidget.itemWidget(item, 0)
                if widget and isinstance(widget, ParticleFileListItem):
                    filepath = widget.filepath
                    hasSelection = False
                    if filepath in self.mainWindow.fileSelectionStates:
                        selection = self.mainWindow.fileSelectionStates[filepath]['selection']
                        hasSelection = len(selection) > 0
                    widget.setHasSelection(hasSelection)
        
        for i in range(self.treeWidget.topLevelItemCount()):
            updateItem(self.treeWidget.topLevelItem(i))
    
    def clearFileSelection(self, filepath):
        """Clear all selections for a specific file"""
        if not self.mainWindow:
            return
        
        if filepath in self.mainWindow.fileSelectionStates:
            self.mainWindow.fileSelectionStates[filepath]['selection'] = []
        
        if self.mainWindow.currentFilePath == filepath:
            self.mainWindow.colorView.clearSelection()
        
        self.updateFileIndicators()
    
    def getGroupStructure(self):
        """Get the tree structure for saving to project file.
        Returns list of: {'type': 'file', 'filepath': ..., 'note': ...}
                     or: {'type': 'group', 'name': ..., 'color': ..., 'children': [...]}
        """
        def itemToDict(item):
            isGroup = item.data(0, self.GROUP_ROLE)
            if isGroup:
                children = []
                for i in range(item.childCount()):
                    children.append(itemToDict(item.child(i)))
                return {
                    'type': 'group',
                    'name': item.text(0).replace("📁 ", ""),
                    'color': item.data(0, self.COLOR_ROLE) or "#4a90d9",
                    'children': children
                }
            else:
                filepath = item.data(0, self.FILEPATH_ROLE)
                idx = self._getFileDataIndex(filepath)
                note = self.fileData[idx][3] if idx >= 0 else ""
                return {
                    'type': 'file',
                    'filepath': filepath,
                    'note': note or ""
                }
        
        result = []
        for i in range(self.treeWidget.topLevelItemCount()):
            result.append(itemToDict(self.treeWidget.topLevelItem(i)))
        return result
    
    def loadGroupStructure(self, structure):
        """Rebuild tree from saved structure"""
        def buildItem(data, parentItem=None):
            if data['type'] == 'group':
                groupItem = QTreeWidgetItem()
                groupItem.setData(0, self.GROUP_ROLE, True)
                groupItem.setData(0, self.COLOR_ROLE, data.get('color', '#4a90d9'))
                groupItem.setText(0, f"📁 {data['name']}")
                groupItem.setForeground(0, QColor(data.get('color', '#4a90d9')))
                
                if parentItem:
                    parentItem.addChild(groupItem)
                else:
                    self.treeWidget.addTopLevelItem(groupItem)
                
                for child in data.get('children', []):
                    buildItem(child, groupItem)
                
                groupItem.setExpanded(True)
                self._groupCounter += 1
            elif data['type'] == 'file':
                filepath = data['filepath']
                if not os.path.exists(filepath):
                    return
                note = data.get('note', '')
                try:
                    with open(filepath, 'rb') as f:
                        fileData = MemoryStream(f.read())
                    effect = ParticleEffect()
                    effect.from_memory_stream(fileData)
                    self.fileData.append([filepath, fileData, effect, note])
                    
                    item = QTreeWidgetItem()
                    item.setData(0, self.GROUP_ROLE, False)
                    item.setData(0, self.FILEPATH_ROLE, filepath)
                    
                    itemWidget = ParticleFileListItem(filepath, effect)
                    itemWidget.clearSelectionClicked.connect(self.clearFileSelection)
                    
                    if parentItem:
                        parentItem.addChild(item)
                    else:
                        self.treeWidget.addTopLevelItem(item)
                    
                    self.treeWidget.setItemWidget(item, 0, itemWidget)
                    item.setSizeHint(0, itemWidget.sizeHint())
                except Exception:
                    pass
        
        for data in structure:
            buildItem(data)
        
    def clear(self):
        self.treeWidget.clear()
        self.fileData.clear()

# ==============================================================================
# MAIN APPLICATION WINDOW
# ==============================================================================

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"HD2 Particle Modder - Version {VERSION}")
        self.setWindowIcon(QIcon("assets/icon.png"))
        self.resize(1100, 700)
        self.particleFilepath = ""
        self.undoStack = QUndoStack(self)

        self.hidden_columns = {
            'color': set(),
            'opacity': set(),
            'size': set()
        }
        # Per-file selection state: {filepath: {'presets': [None]*2, 'selection': [(row,col)]}}
        self.fileSelectionStates = {}
        self.currentFilePath = None
        self._isLoadingFile = False  # Flag to prevent saving selection during file load
        self.setStatusBar(QStatusBar(self))
        self.initComponents()
        self.filenameLabel = QLabel("No file loaded")
        self.connectComponents()
        self.layoutComponents()

    def initComponents(self):
        self.initMenuBar()
        self.initFilesWindow()
        self.initTabWidget()
        self.initMaterialView()
        self.initColorView()
        self.initOpacityView()
        self.initLifetimeView()
        self.initSizeView()
        #self.initPositionView()
        #self.initRotationView()

    def connectComponents(self):
        self.fileOpenArchiveAction.triggered.connect(self.load_archive)
        self.fileSaveAsAction.triggered.connect(self.saveArchive)
        self.fileSaveAllFilesAction.triggered.connect(self.saveProjectFiles)
        self.fileSaveArchiveAction.triggered.connect(self.saveSelectedFile)
        self.fileSaveProjectAction.triggered.connect(self.saveProject)
        self.fileLoadProjectAction.triggered.connect(self.loadProject)
        self.fileCloseAllAction.triggered.connect(self.closeAllFiles)
        self.fileCloseAction.triggered.connect(self.closeCurrentFile)
        
        self.loadedFilesStrip.loadFile.connect(self.loadFromStream)

    def layoutComponents(self):
        self.setMinimumSize(300, 200)
        self.layout = QVBoxLayout()
        
        self.splitter = QSplitter(Qt.Horizontal)
        #self.splitter.addWidget(self.colorView)

        # Floating header strip widget
        filenameStrip = QWidget(self)
        filenameStripLayout = QHBoxLayout()
        filenameStripLayout.setContentsMargins(8, 4, 8, 4)

        self.filenameLabel.setText("No file loaded")
        self.filenameLabel.setStyleSheet("font-weight: bold; font-size: 12px; color: white; text-decoration: none;")

        self.openFileBtn = QToolButton(self)
        self.openFileBtn.setText("Open")
        self.openFileBtn.clicked.connect(lambda: self.load_archive())

        self.saveFileBtn = QToolButton(self)
        self.saveFileBtn.setText("Save")
        self.saveFileBtn.clicked.connect(lambda: self.saveArchive())

        filenameStripLayout.addWidget(self.filenameLabel)
        filenameStripLayout.addStretch()
        filenameStripLayout.addWidget(self.openFileBtn)
        filenameStripLayout.addWidget(self.saveFileBtn)

        filenameStrip.setLayout(filenameStripLayout)
        filenameStrip.setStyleSheet("""
            background-color: #434343;
        """)
        filenameStrip.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        
        self.layout.addWidget(filenameStrip)
        self.layout.addWidget(self.loadedFilesStrip)
        #self.splitter.addWidget(self.loadedFilesWindow)
        #self.splitter.addWidget(self.particleEffectView)
        #self.splitter.addWidget(self.tabWidget)
        self.layout.addWidget(self.tabWidget)
        #self.layout.addWidget(self.splitter)
        #self.layout.addStretch()
        #self.layout.addWidget(self.tabWidget)

        # Color tab layout
        layout = QVBoxLayout()
        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(self.hideTimeColumnsBtn)
        self.hideTimeColumnsBtn.setToolTip("Toggle visibility of time columns")
        buttonLayout.addWidget(self.pickColorBtn)
        
        # Add separator
        buttonLayout.addSpacing(20)
        
        # Add preset buttons (Save P1-P2)
        for i in range(2):
            saveBtn = QToolButton(self.colorTab)
            saveBtn.setText(f"Save P{i+1}")
            saveBtn.setToolTip(f"Save current selection to Preset {i+1}")
            saveBtn.clicked.connect(lambda checked=False, idx=i: self.saveColorSelectionPreset(idx))
            buttonLayout.addWidget(saveBtn)
        
        buttonLayout.addSpacing(10)
        
        # Add preset buttons (Load P1-P2)
        for i in range(2):
            loadBtn = QToolButton(self.colorTab)
            loadBtn.setText(f"Load P{i+1}")
            loadBtn.setToolTip(f"Load selection from Preset {i+1}")
            loadBtn.clicked.connect(lambda checked=False, idx=i: self.loadColorSelectionPreset(idx))
            buttonLayout.addWidget(loadBtn)
        
        buttonLayout.addStretch()
        layout.addLayout(buttonLayout)
        layout.addWidget(self.colorView)
        self.colorTab.setLayout(layout)

        # Opacity tab layout
        layout = QVBoxLayout()
        opacityButtonLayout = QHBoxLayout()
        opacityButtonLayout.addWidget(self.hideOpacityTimeColumnsBtn)
        self.hideOpacityTimeColumnsBtn.setToolTip("Toggle visibility of time columns")
        layout.addLayout(opacityButtonLayout)
        layout.addWidget(self.opacityView)
        self.opacityTab.setLayout(layout)

        # Lifetime tab layout
        layout = QVBoxLayout()
        layout.addWidget(self.lifetimeView)
        self.lifetimeTab.setLayout(layout)

        # Size Scale tab layout
        layout = QVBoxLayout()
        sizeButtonLayout = QHBoxLayout()
        sizeButtonLayout.addWidget(self.hideSizeTimeColumnsBtn)
        self.hideSizeTimeColumnsBtn.setToolTip("Toggle visibility of time columns")
        layout.addLayout(sizeButtonLayout)
        layout.addWidget(self.sizeView)
        self.sizeTab.setLayout(layout)
        
        # Visualizer tab layout
        layout = QVBoxLayout()
        layout.addWidget(self.particleMaterialView)
        self.materialTab.setLayout(layout)

        #layout = QVBoxLayout()
        #layout.addWidget(self.positionView)
        #self.positionTab.setLayout(layout)

        #layout = QVBoxLayout()
        #layout.addWidget(self.rotationView)
        #self.rotationTab.setLayout(layout)

        self.tabWidget.addTab(self.colorTab, "Color")
        self.tabWidget.addTab(self.opacityTab, "Opacity")
        self.tabWidget.addTab(self.sizeTab, "Intensity")
        self.tabWidget.addTab(self.lifetimeTab, "Lifetime")
        #self.tabWidget.addTab(self.positionTab, "Emitter Offset")
        #self.tabWidget.addTab(self.rotationTab, "Emitter Rotation")
        self.tabWidget.addTab(self.materialTab, "Visualizers")

        widget = QWidget()
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)
        
    def initMaterialView(self):
        self.particleMaterialView = ParticleMaterialView(self)
        
    def initParticleView(self):
        self.particleEffectView = ParticleEffectView(self)
        
    def initFilesWindow(self):
        self.loadedFilesStrip = LoadedFilesBar(self)
        self.loadedFilesStrip.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)

    def initColorView(self):
        self.colorView = ColorTable(self)
        self.colorViewModel = ColorGradientModel(self.undoStack)
        self.colorView.setModel(self.colorViewModel)

        delegate = ColorSwatchDelegate()
        self.colorView.setItemDelegate(delegate)
        
        # Track selection changes to update indicators
        self.colorView.selectionModel().selectionChanged.connect(self.onColorSelectionChanged)

        self.pickColorBtn = QToolButton(self.colorTab)
        self.pickColorBtn.setText("Color Picker")
        self.pickColorBtn.setToolTip("Open Color Picker for selected color cell")
        self.pickColorBtn.clicked.connect(self.colorView.triggerColorPickerFromButton)

        self.hideTimeColumnsBtn = QToolButton(self.colorTab)
        self.hideTimeColumnsBtn.setText("Toggle Time Columns")
        self.hideTimeColumnsBtn.clicked.connect(self.toggleTimeColumns)
    
    def onColorSelectionChanged(self):
        """Called when color cell selection changes"""
        # Don't save during file loading/restoring
        if self._isLoadingFile:
            return
        # Save current selection state
        self.saveCurrentSelectionState()
        # Update file indicators to show which files have selections
        if hasattr(self, 'loadedFilesStrip'):
            self.loadedFilesStrip.updateFileIndicators()

    def toggleTimeColumns(self):
        for col in range(self.colorViewModel.columnCount()):
            header = self.colorViewModel.headerData(col, Qt.Horizontal)
            if isinstance(header, str) and header.lower().startswith("time"):
                hidden = self.colorView.isColumnHidden(col)
                self.colorView.setColumnHidden(col, not hidden)
                if not hidden:
                    self.hidden_columns['color'].add(col)
                else:
                    self.hidden_columns['color'].discard(col)

    def initOpacityView(self):
        self.opacityView = OpacityTable(self)
        self.opacityViewModel = OpacityGradientModel(self.undoStack)
        self.opacityView.setModel(self.opacityViewModel)

        self.hideOpacityTimeColumnsBtn = QToolButton(self.opacityTab)
        self.hideOpacityTimeColumnsBtn.setText("Toggle Time Columns")
        self.hideOpacityTimeColumnsBtn.clicked.connect(self.toggleOpacityTimeColumns)

    def toggleOpacityTimeColumns(self):
        for col in range(self.opacityViewModel.columnCount()):
            header = self.opacityViewModel.headerData(col, Qt.Horizontal)
            if isinstance(header, str) and header.lower().startswith("time"):
                hidden = self.opacityView.isColumnHidden(col)
                self.opacityView.setColumnHidden(col, not hidden)
                if not hidden:
                    self.hidden_columns['opacity'].add(col)
                else:
                    self.hidden_columns['opacity'].discard(col)
    
    def saveColorSelectionPreset(self, presetIndex):
        """Save current color cell selection to preset slot (0-3)"""
        if not self.currentFilePath:
            return
        selectedIndexes = self.colorView.selectedIndexes()
        if not selectedIndexes:
            return
        # Store as list of (row, col) tuples
        preset = [(idx.row(), idx.column()) for idx in selectedIndexes]
        if self.currentFilePath not in self.fileSelectionStates:
            self.fileSelectionStates[self.currentFilePath] = {'presets': [None]*2, 'selection': []}
        self.fileSelectionStates[self.currentFilePath]['presets'][presetIndex] = preset
    
    def loadColorSelectionPreset(self, presetIndex):
        """Load color cell selection from preset slot (0-3)"""
        if not self.currentFilePath or self.currentFilePath not in self.fileSelectionStates:
            return
        preset = self.fileSelectionStates[self.currentFilePath]['presets'][presetIndex]
        if preset is None:
            return
        
        # Clear current selection
        self.colorView.clearSelection()
        
        # Select cells from preset
        selectionModel = self.colorView.selectionModel()
        for row, col in preset:
            index = self.colorViewModel.index(row, col)
            if index.isValid():
                selectionModel.select(index, QItemSelectionModel.Select)
    
    def saveCurrentSelectionState(self):
        """Save current color cell selection state before switching files"""
        if not self.currentFilePath:
            return
        selectedIndexes = self.colorView.selectedIndexes()
        selection = [(idx.row(), idx.column()) for idx in selectedIndexes]
        
        if self.currentFilePath not in self.fileSelectionStates:
            self.fileSelectionStates[self.currentFilePath] = {'presets': [None]*2, 'selection': []}
        self.fileSelectionStates[self.currentFilePath]['selection'] = selection
    
    def restoreSelectionState(self):
        """Restore color cell selection state for current file"""
        if not self.currentFilePath:
            return
        
        # Clear current selection
        self.colorView.clearSelection()
        
        # Restore if exists
        if self.currentFilePath in self.fileSelectionStates:
            selection = self.fileSelectionStates[self.currentFilePath]['selection']
            selectionModel = self.colorView.selectionModel()
            for row, col in selection:
                index = self.colorViewModel.index(row, col)
                if index.isValid():
                    selectionModel.select(index, QItemSelectionModel.Select)

    def initSizeView(self):
        self.sizeView = QTableView(self)
        self.sizeViewModel = SizeModel(self.undoStack)
        self.sizeView.setModel(self.sizeViewModel)

        self.hideSizeTimeColumnsBtn = QToolButton(self.sizeTab)
        self.hideSizeTimeColumnsBtn.setText("Toggle Time Columns")
        self.hideSizeTimeColumnsBtn.clicked.connect(self.toggleSizeTimeColumns)

    def toggleSizeTimeColumns(self):
        for col in range(self.sizeViewModel.columnCount()):
            header = self.sizeViewModel.headerData(col, Qt.Horizontal)
            if isinstance(header, str) and header.lower().startswith("time"):
                hidden = self.sizeView.isColumnHidden(col)
                self.sizeView.setColumnHidden(col, not hidden)
                if not hidden:
                    self.hidden_columns['size'].add(col)
                else:
                    self.hidden_columns['size'].discard(col)

    def applyHiddenColumns(self, key, tableView):
        for col in range(tableView.model().columnCount()):
            tableView.setColumnHidden(col, col in self.hidden_columns[key])

    def initLifetimeView(self):
        self.lifetimeView = QTableView(self)
        self.lifetimeViewModel = LifetimeModel()
        self.lifetimeView.setModel(self.lifetimeViewModel)

    def initPositionView(self):
        self.positionView = QTableView(self)
        self.positionViewModel = PositionModel()
        self.positionView.setModel(self.positionViewModel)

    def initRotationView(self):
        self.rotationView = QTableView(self)
        self.rotationViewModel = RotationModel()
        self.rotationView.setModel(self.rotationViewModel)

    def initTabWidget(self):
        self.tabWidget = QTabWidget(self)
        self.colorTab = QWidget(self.tabWidget)
        self.opacityTab = QWidget(self.tabWidget)
        self.lifetimeTab = QWidget(self.tabWidget)
        self.sizeTab = QWidget(self.tabWidget)
        #self.positionTab = QWidget(self.tabWidget)
        #self.rotationTab = QWidget(self.tabWidget)
        self.materialTab = QWidget(self.tabWidget)

    def initMenuBar(self):
        menu_bar = self.menuBar()

        self.file_menu = menu_bar.addMenu("File")

        self.fileOpenArchiveAction = QAction("Open", self)
        self.fileOpenArchiveAction.setShortcut(QKeySequence.Open)
        self.fileSaveArchiveAction = QAction("Save", self)
        self.fileSaveArchiveAction.setShortcut(QKeySequence.Save)
        self.fileSaveAsAction =      QAction("Save As", self)
        self.fileSaveAsAction.setShortcut(QKeySequence.SaveAs)
        self.fileSaveAllFilesAction= QAction("Save All", self)
        self.fileLoadProjectAction = QAction("Open Project File", self)
        self.fileSaveProjectAction = QAction("Save Project File", self)
        self.fileCloseAction =       QAction("Close File", self)
        self.fileCloseAction.setShortcut(QKeySequence("Ctrl+W"))
        self.fileCloseAllAction =    QAction("Close All", self)

        self.file_menu.addAction(self.fileOpenArchiveAction)
        self.file_menu.addAction(self.fileLoadProjectAction)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.fileSaveArchiveAction)
        self.file_menu.addAction(self.fileSaveAsAction)
        self.file_menu.addAction(self.fileSaveAllFilesAction)
        self.file_menu.addAction(self.fileSaveProjectAction)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.fileCloseAction)
        self.file_menu.addAction(self.fileCloseAllAction)

        self.edit_menu = menu_bar.addMenu("Edit")
        self.undo_action = self.undoStack.createUndoAction(self, "Undo")
        self.undo_action.setShortcut(QKeySequence.Undo)
        self.redo_action = self.undoStack.createRedoAction(self, "Redo")
        self.redo_action.setShortcut(QKeySequence.Redo)
        self.edit_menu.addAction(self.undo_action)
        self.edit_menu.addAction(self.redo_action)
        
    def loadFromStream(self, filepath: str, stream: MemoryStream, particleEffect: ParticleEffect):
        # Save current selection state before switching
        if not self._isLoadingFile:
            self.saveCurrentSelectionState()
        
        # Set flag to prevent selection changes during load
        self._isLoadingFile = True
        
        self.particleEffectData = stream
        self.particleEffect = particleEffect
        self.currentFilePath = filepath
        self.reloadData()
        self.setLoadedFileLabels(filepath)
        
        # Restore selection state for new file
        self.restoreSelectionState()
        
        # Clear flag
        self._isLoadingFile = False
        
        # Update file list indicators
        self.loadedFilesStrip.updateFileIndicators()
        
    def setLoadedFileLabels(self, filepath):
        self.statusBar().showMessage(f"Loaded: {os.path.basename(filepath)}", 5000)
        stat = os.stat(filepath)
        modified_time = time.strftime('%Y-%m-%d %H:%M', time.localtime(stat.st_mtime))
        self.name = os.path.basename(filepath)
        self.particleFilepath = filepath
        self.filenameLabel.setText(f"{os.path.basename(filepath)} — last modified: {modified_time}")
        
    def reloadData(self):
        self.particleMaterialView.loadData(self.particleEffect)
        self.colorViewModel.setParticleEffect(self.particleEffect)
        self.opacityViewModel.setParticleEffect(self.particleEffect)
        self.lifetimeViewModel.setParticleEffect(self.particleEffect)
        self.sizeViewModel.setParticleEffect(self.particleEffect)
        self.applyHiddenColumns('color', self.colorView)
        self.applyHiddenColumns('opacity', self.opacityView)
        self.applyHiddenColumns('size', self.sizeView)
                
    def saveProject(self, initialdir: str | None = '', outputFile: str | None = ""):
        import json
        if not outputFile:
            outputFile = QFileDialog.getSaveFileName(self, "Save File", str(initialdir), "Particle Mod (*.pmod)")
            outputFile = outputFile[0]
        if not outputFile:
            return
        
        # Save current file's selection before serializing
        self.saveCurrentSelectionState()
        
        # Get tree structure with groups
        structure = self.loadedFilesStrip.getGroupStructure()
        
        # Serialize selection states
        selectionStates = {}
        for fp, state in self.fileSelectionStates.items():
            selectionStates[fp] = {
                'selection': state.get('selection', []),
                'presets': [
                    preset if preset is not None else None
                    for preset in state.get('presets', [None, None])
                ]
            }
        
        projectData = {
            'version': 2,
            'project_name': 'default project',
            'structure': structure,
            'selectionStates': selectionStates
        }
        
        with open(outputFile, 'w', encoding='utf-8') as f:
            json.dump(projectData, f, indent=2, ensure_ascii=False)
        
        self.statusBar().showMessage(f"Project saved to {os.path.basename(outputFile)}", 3000)
        
    def saveProjectFiles(self):
        projectFiles = self.loadedFilesStrip.getAllLoadedFiles()
        for item in projectFiles:
            path, stream, particleEffect, note = item
            stream.seek(0)
            particleEffect.write_to_memory_stream(stream)
            with open(path, 'wb') as f:
                f.write(stream.data)
        self.statusBar().showMessage(f"Saved all particle files", 3000)
        
    def loadProject(self, initialdir: str | None = '', projectFile: str | None = ""):
        import json
        if not projectFile:
            projectFile = QFileDialog.getOpenFileName(self, "Select Project File", str(initialdir), "Particle Mod (*.pmod)")
            projectFile = projectFile[0]
        if not projectFile:
            return
        self.closeAllFiles()
        
        # Try JSON format (v2) first, then fall back to XML (v1)
        try:
            with open(projectFile, 'r', encoding='utf-8') as f:
                projectData = json.load(f)
            
            structure = projectData.get('structure', [])
            self.loadedFilesStrip.loadGroupStructure(structure)
            
            # Restore selection states
            savedStates = projectData.get('selectionStates', {})
            for fp, state in savedStates.items():
                self.fileSelectionStates[fp] = {
                    'selection': [tuple(s) for s in state.get('selection', [])],
                    'presets': [
                        [tuple(s) for s in preset] if preset is not None else None
                        for preset in state.get('presets', [None, None])
                    ]
                }
            
            # Update indicators and restore current file selection
            self.loadedFilesStrip.updateFileIndicators()
            if self.currentFilePath:
                self.restoreSelectionState()
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Fall back to old XML format
            tree = ET.parse(projectFile)
            root = tree.getroot()
            for project in root:
                projectFiles = project.find('project_files')
                for file in projectFiles:
                    filepath = file.find('filepath').text
                    if not os.path.exists(filepath):
                        continue
                    note = file.find('note').text
                    with open(filepath, 'rb') as f:
                        fileData = MemoryStream(f.read())
                    particleEffect = ParticleEffect()
                    particleEffect.from_memory_stream(fileData)
                    self.addLoadedFile(filepath, fileData, particleEffect, note)
                break
            
    def closeAllFiles(self):
        self.loadedFilesStrip.clear()
        
    def closeCurrentFile(self):
        self.loadedFilesStrip.closeCurrentTab()
    
    def addLoadedFile(self, filepath: str, fileData: MemoryStream, particleEffect: ParticleEffect, note: str=""):
        self.loadedFilesStrip.addFile(filepath, fileData, particleEffect, note)
        #self.loadedFilesWindow.addFile(filepath, fileData, particleEffect, note)

    def load_archive(self, initialdir: str | None = '', archive_file: str | None = ""):
        archive_files = []
        if not archive_file:
            # Use getOpenFileNames (plural) to allow multiple file selection
            archive_files_result = QFileDialog.getOpenFileNames(self, "Select archive(s)", str(initialdir), "Particle Files (*.particles *.pmod);;All Files (*.*)")
            archive_files = archive_files_result[0]
        else:
            # Single file provided programmatically
            archive_files = [archive_file]
            
        if not archive_files:
            return
        
        # Process each selected file
        for idx, current_file in enumerate(archive_files):
            if os.path.splitext(current_file)[1] == ".pmod":
                self.loadProject(projectFile = current_file)
                continue
                
            self.name = current_file
            with open(current_file, "rb") as f:
                particleEffectData = MemoryStream(f.read())
            particleEffect = ParticleEffect()
            particleEffect.from_memory_stream(particleEffectData)
            
            # Only set as current effect for the first file or last file
            if idx == len(archive_files) - 1:
                self.particleEffectData = particleEffectData
                self.particleEffect = particleEffect
                self.reloadData()
                self.setLoadedFileLabels(current_file)
                
                # Reapply hidden column states for the last file
                self.applyHiddenColumns('color', self.colorView)
                self.applyHiddenColumns('opacity', self.opacityView)
                self.applyHiddenColumns('size', self.sizeView)
            
            self.addLoadedFile(current_file, particleEffectData, particleEffect)
            
        #self.positionViewModel.setFileData(self.data)
        #self.rotationViewModel.setFileData(self.data)
        
        
        
    def saveArchive(self, initialdir: str | None = '', archive_file: str | None = ""):
        saveAs = False
        if not archive_file: # save-as operation
            saveAs = True
            archive_file = QFileDialog.getSaveFileName(self, "Select archive", self.particleFilepath)
            archive_file = archive_file[0]
        if not archive_file:
            return
        with open(archive_file, "wb") as f:
            self.particleEffectData.seek(0)
            self.particleEffect.write_to_memory_stream(self.particleEffectData)
            f.write(self.particleEffectData.data)
            #data = MemoryStream()
            #data.write(self.data)
            #self.colorViewModel.writeFileData(data)
            #self.lifetimeViewModel.writeFileData(data)
            #self.opacityViewModel.writeFileData(data)
            #self.sizeViewModel.writeFileData(data)
            #self.positionViewModel.writeFileData(data)
            #self.rotationViewModel.writeFileData(data)
            #f.write(data.data)
            self.statusBar().showMessage(f"Saved: {os.path.basename(archive_file)}", 5000)
        if saveAs:
            self.loadedFilesStrip.setCurrentFilePath(archive_file)
            self.setLoadedFileLabels(archive_file)
            
    def saveSelectedFile(self):
        file = self.loadedFilesStrip.getSelectedFile()
        if file:
            self.saveArchive(archive_file=file[0])

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            filename = url.toLocalFile()
            if os.path.isfile(filename):
                self.load_archive(archive_file=filename)

    def dragEnterEvent(self, event):
        for url in event.mimeData().urls():
            if not os.path.isfile(url.toLocalFile()):
                event.ignore()
                return
        event.accept()

    def dragMoveEvent(self, event):
        for url in event.mimeData().urls():
            if not os.path.isfile(url.toLocalFile()):
                event.ignore()
                return
        event.accept()

def get_dark_mode_palette( app=None ):

    darkPalette = app.palette()
    darkPalette.setColor( QPalette.Window, QColor( 53, 53, 53 ) )
    darkPalette.setColor( QPalette.WindowText, Qt.white )
    darkPalette.setColor( QPalette.Disabled, QPalette.WindowText, QColor( 127, 127, 127 ) )
    darkPalette.setColor( QPalette.Base, QColor( 42, 42, 42 ) )
    darkPalette.setColor( QPalette.AlternateBase, QColor( 66, 66, 66 ) )
    darkPalette.setColor( QPalette.ToolTipBase, QColor( 53, 53, 53 ) )
    darkPalette.setColor( QPalette.ToolTipText, Qt.white )
    darkPalette.setColor( QPalette.Text, Qt.white )
    darkPalette.setColor( QPalette.Disabled, QPalette.Text, QColor( 127, 127, 127 ) )
    darkPalette.setColor( QPalette.Dark, QColor( 35, 35, 35 ) )
    darkPalette.setColor( QPalette.Shadow, QColor( 20, 20, 20 ) )
    darkPalette.setColor( QPalette.Button, QColor( 53, 53, 53 ) )
    darkPalette.setColor( QPalette.ButtonText, Qt.white )
    darkPalette.setColor( QPalette.Disabled, QPalette.ButtonText, QColor( 127, 127, 127 ) )
    darkPalette.setColor( QPalette.BrightText, Qt.red )
    darkPalette.setColor( QPalette.Link, QColor( 42, 130, 218 ) )
    darkPalette.setColor( QPalette.Highlight, QColor( 42, 130, 218 ) )
    darkPalette.setColor( QPalette.Disabled, QPalette.Highlight, QColor( 80, 80, 80 ) )
    darkPalette.setColor( QPalette.HighlightedText, Qt.white )
    darkPalette.setColor( QPalette.Disabled, QPalette.HighlightedText, QColor( 127, 127, 127 ), )

    return darkPalette

# ==============================================================================
# APPLICATION ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    app = QApplication([])
    app.setStyle("Fusion")
    app.setPalette(get_dark_mode_palette(app))
    graphs_set_dark_mode()

    window = MainWindow()

    window.show()

    app.exec()