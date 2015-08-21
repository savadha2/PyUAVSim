# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 15:57:56 2015

@author: sharath
"""
import mpl_toolkits.mplot3d as a3
import numpy as np

class UAVViewer:
    R_nedtoplot = np.matrix(([0, 1, 0], [1, 0, 0], [0, 0, -1]), dtype = np.double)
    def __init__(self, ax, vertices, faces, colors):
        vertices = np.array(vertices * self.R_nedtoplot.T)
        self.assemblies = []
        self.faces = faces
        for k, face in enumerate(self.faces):
            self.assemblies.append(self.add_component(vertices[face, :], face_color = colors[k]))        
        self.ax = ax
        self.draw()
        
    def add_component(self, faces, face_color = 'b', edge_color = 'k'):
        poly_collections = []
        for face in faces:
            poly_collection = a3.art3d.Poly3DCollection([face])
            poly_collection.set_color(face_color)#colors.rgb2hex(sp.rand(3)))
            poly_collection.set_edgecolor(edge_color)
            poly_collections.append(poly_collection)
        return poly_collections

    def draw_assembly(self, assembly):
        for a in assembly:
            self.ax.add_collection(a)
            
    def update_assembly(self, vertices, assembly, faces):
        for i in range(len(assembly)):
            assembly[i].set_verts([vertices[faces[i], :]])
            
    def draw(self):
        for assembly in self.assemblies:
            self.draw_assembly(assembly)      
        
    def update(self, vertices):
        vertices = np.array(vertices * self.R_nedtoplot.T)
        for k, assembly in enumerate(self.assemblies):
            self.update_assembly(vertices, assembly, self.faces[k])