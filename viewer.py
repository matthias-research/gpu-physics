# Copyright 2023 Matthias Müller - Ten Minute Physics, 
# www.youtube.com/c/TenMinutePhysics
# www.matthiasMueller.info/tenMinutePhysics

# MIT License

# Permission is hereby granted, free of charge, to any person obtaining a 
# copy of this software and associated documentation files (the "Software"), 
# to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, 
# and/or sell copies of the Software, and to permit persons to whom the Software 
# is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
# THE SOFTWARE.

# ----------------------------------------------------------------------------------
# GLUT based interactive viewer
# ----------------------------------------------------------------------------------

# install imageio: pip install imageio

# install warp: pip install warp-lang

# install PyOpenGL and PyOpenGL_accelerate: 

# does not work from pip for me
# download PyOpenGL*.whl and PyOpenGL_accelerate*.whl from https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopengl
# pick the right version for your python version and 32/64 bit
# pip install *.whl


import numpy as np
import math
import time
import imageio

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.GL import shaders

keyStrings = {
    GLUT_KEY_DOWN: b"down",
    GLUT_KEY_UP: b"up",
    GLUT_KEY_LEFT: b"left",
    GLUT_KEY_RIGHT: b"right",    
}

cameraFieldOfView = 60.0
cameraNear = 0.1
cameraFar = 3000.0
lightNear = 1.0
lightFar = 1000.0

renderGroundPlane = True
renderAxes = True
paused = True
singleStep = False
renderShadows = True
renderLighting = True
renderDepthMap = False

# globals

camera = None
renderer = None
userApp = None

targetFps = 60
mouseButton = 0
mouseX = 0
mouseY = 0
shiftDown = False
prevTime = time.time()
frameNr = 0

depthResolution = 1024

quadric = None


# ----------------------------------------------------------------------------------
def addMesh(mesh):

    if renderer:
        renderer.addMesh(mesh)


def addParticles(particles):

    if renderer:
        renderer.addParticles(particles)


def createDefaultLights(distance = 5.0):
    
    if renderer:
        renderer.createDefaultLights(distance)  


# ----------------------------------------------------------------------------------
class Mesh:

    def __init__(self):

        self.verts = []
        self.triIds = []
        self.normals = []
        self.uvs = []

        self.color = np.array([1.0, 1.0, 1.0])
        self.transform = np.identity(4, dtype=np.float32)

        self.vertexArrayObject = 0
        self.vertexBuffer = 0
        self.normalBuffer = 0
        self.uvBuffer = 0
        self.indexBuffer = 0
        self.textureId = 0


    def createBuffers(self):

        if len(self.verts) == 0:
            return
        
        self.vertexArrayObject = glGenVertexArrays(1)
        glBindVertexArray(self.vertexArrayObject)

        self.vertexBuffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertexBuffer)
        glBufferData(GL_ARRAY_BUFFER, self.verts, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        if len(self.normals) == len(self.verts):
            self.normalBuffer = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.normalBuffer)
            glBufferData(GL_ARRAY_BUFFER, self.normals, GL_STATIC_DRAW)
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
            glEnableVertexAttribArray(1)

        if len(self.uvs) == len(self.verts):
            self.uvBuffer = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.uvBuffer)
            glBufferData(GL_ARRAY_BUFFER, self.uvs, GL_STATIC_DRAW)
            glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, None)
            glEnableVertexAttribArray(2)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        self.indexBuffer = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.indexBuffer)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.triIds, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)


    def loadTexture(self, path):

        try:
            image = imageio.imread(path)
        except:
            return False
            
        colorType = GL_RGB if image.shape[2] == 3 else GL_RGBA
        image = np.flipud(image)

        # image[image < 50] = 200
        # imageio.imwrite("out.png", image)
        
        self.textureId = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.textureId)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexImage2D(GL_TEXTURE_2D, 0, colorType, image.shape[1], image.shape[0], 0, colorType, GL_UNSIGNED_BYTE, image)
        glBindTexture(GL_TEXTURE_2D, 0)

        return True


    def loadObjFile(self, filePath, texturePath = None):

        objVerts = []
        objUvs = []
        objNormals = []
        vertDict = {}

        verts = []
        uvs = []
        normals = []
        triIds = []

        hasNormals = False
        hasUvs = False

        try:
            fin = open(filePath, 'r')
        except:
            return False
        
        for line in fin:
            fields = line.split()
            if len(fields) < 2: 
                continue

            if fields[0] == "v":
                coords = [float(i) for i in fields[1:4]]
                objVerts.append(coords)

            if fields[0] == "vt":
                uv = [float(i) for i in fields[1:3]]
                objUvs.append(uv)
                hasUvs = True

            if fields[0] == "vn":
                normal = [float(i) for i in fields[1:4]]
                objNormals.append(normal)
                hasNormals = True

            if fields[0] == "f":
                faceIds = []

                for i in range(1, len(fields)):
                    field = fields[i]
                    vertNr = vertDict.get(field)
                    if vertNr is None:
                        vertDict[field] = len(verts)
                        vertNr = len(verts)
                        ids = fields[i].split("/")
                        verts.append(objVerts[int(ids[0]) - 1])
                        if ids[1] == '':
                            uvs.append([0.0, 0.0])
                        else:
                            uvs.append(objUvs[int(ids[1]) - 1])
                        if ids[2] == '':
                            normals.append([0.0, 0.0, 0.0])
                        else:
                            normals.append(objNormals[int(ids[2]) - 1])
                    faceIds.append(vertNr)

                for i in range(1, len(faceIds) - 1):
                    triIds.append(faceIds[0])
                    triIds.append(faceIds[i])
                    triIds.append(faceIds[i + 1])

        fin.close() 
        self.verts = np.array(verts, dtype=np.float32)
        self.normals = np.array(normals, dtype=np.float32)
        if hasUvs:
            self.uvs = np.array(uvs, dtype=np.float32)
        self.triIds = np.array(triIds, dtype=int)

        if not hasNormals:
            self.updateNormals()

        self.createBuffers()
        if texturePath is not None:
            self.loadTexture(texturePath)

        return True
   

    def updateNormals(self):

        ids = np.reshape(self.triIds, (-1, 3) )            
        triN = np.cross(self.verts[ids[:,1]] - self.verts[ids[:,0]], self.verts[ids[:,2]] - self.verts[ids[:,0]])
        self.normals = np.zeros(self.verts.shape, np.float32)
        self.normals[ids[:,0]] += triN[:]
        self.normals[ids[:,1]] += triN[:]
        self.normals[ids[:,2]] += triN[:]
        self.normals /= np.sqrt(np.sum(self.normals**2, axis=1, keepdims=True))


# ----------------------------------------------------------------------------------

class Particles:

    def __init__(self, pos, quats, radius, radii=None):

        self.pos = pos
        self.radius = radius
        self.quats = quats
        self.radii = radii

        self.vertexArrayObject = 0
        self.posBuffer = 0
        self.quatsBuffer = 0
        self.radiiBuffer = 0

        if radii is not None:
            self.radius = 0.0
        elif radius == 0.0:
            radius = 0.01   # radius > 0.0 means radii not provided

        self.createBuffers()


    def createBuffers(self):
        
        self.vertexArrayObject = glGenVertexArrays(1)
        glBindVertexArray(self.vertexArrayObject)

        self.posBuffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.posBuffer)
        glBufferData(GL_ARRAY_BUFFER, self.pos, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        if self.quats is not None:
            self.quatsBuffer = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.quatsBuffer)
            glBufferData(GL_ARRAY_BUFFER, self.quats, GL_STATIC_DRAW)
            glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, None)
            glEnableVertexAttribArray(1)

        if self.radii is not None:
            self.radiiBuffer = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.radiiBuffer)
            glBufferData(GL_ARRAY_BUFFER, self.radii, GL_STATIC_DRAW)
            glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, None)
            glEnableVertexAttribArray(2)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)


    def update(self, pos, quats=None):

        glBindBuffer(GL_ARRAY_BUFFER, self.posBuffer)
        glBufferData(GL_ARRAY_BUFFER, pos, GL_DYNAMIC_DRAW)

        if quats is not None:
            glBindBuffer(GL_ARRAY_BUFFER, self.quatsBuffer)
            glBufferData(GL_ARRAY_BUFFER, quats, GL_DYNAMIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        


# ----------------------------------------------------------------------------------

def projectionMatrix(fov, aspectRatio, near, far):

    f = 1 / np.tan(np.radians(fov) / 2)
    return np.array([
        [f / aspectRatio, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0]])


def orthographicMatrix(left, right, bottom, top, zNear, zFar):

    rl = right - left
    tb = top - bottom
    fn = zFar - zNear

    tx = - (right + left) / (right - left)
    ty = - (top + bottom) / (top - bottom)
    tz = - (zFar + zNear) / (zFar - zNear)

    return np.array([
        [2.0 / rl, 0.0,      0.0,       tx],
        [0.0,      2.0 / tb, 0.0,       ty],
        [0.0,      0.0,      -2.0 / fn, tz],
        [0.0,      0.0,      0.0,       1.0]
    ], dtype=np.float32)


def lookAtMatrix(eye, target, up):
    z = eye - target
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    y = y / np.linalg.norm(y)

    M = np.identity(4, dtype=np.float32)
    M[0, 0:3] = x
    M[1, 0:3] = y
    M[2, 0:3] = z
    M[0, 3] = -np.dot(x, eye)
    M[1, 3] = -np.dot(y, eye)
    M[2, 3] = -np.dot(z, eye)
    return M

# ----------------------------------------------------------------------------------

class Light:

    def __init__(self, pos, at, range, ):

        self.pos = np.array(pos)
        self.color = np.array([1.0, 1.0, 1.0])
        at = np.array(at)
        self.dir = at - self.pos
        self.dir /= np.linalg.norm(self.dir)
        self.range = range
        self.viewMatrix = np.identity(4, dtype=np.float32)
        self.viewMatrix = lookAtMatrix(self.pos, at, np.array([0.0, 1.0, 0.0]))
        self.depthFBO = 0
        self.depthTexture = 0

    def update(self, pos, at, range):
            
            self.range = range
            self.pos = np.array(pos)
            at = np.array(at)
            self.dir = at - self.pos
            self.dir /= np.linalg.norm(self.dir)
            self.viewMatrix = np.identity(4, dtype=np.float32)
            self.viewMatrix = lookAtMatrix(self.pos, at, np.array([0.0, 1.0, 0.0]))


    def createDepthMap(self, width, height):
            
        self.depthTexture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.depthTexture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 
                    width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)  
                
        self.depthFBO = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.depthFBO)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.depthTexture, 0)

        glDrawBuffer(GL_NONE)
        glReadBuffer(GL_NONE)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)


# ----------------------------------------------------------------------------------


class Renderer:

    def __init__(self):

        self.defaultShader = None
        self.spriteShader = None
        self.flatShader = None
        self.depthDisplayShader = None
        self.particles = []
        self.meshes = []
        self.lights = []

        self.groundVerts = None
        self.groundIds = None
        self.groundColors = None

        self.axesArrayObject = 0

        self.quadVAO = 0
        self.quadVBO = 0

        self.projectionMatrix = np.identity(4, dtype=np.float32)
        self.viewMatrix = np.identity(4, dtype=np.float32)
        self.viewPos = np.zeros(3, dtype=np.float32)
        self.lightProjectionMatrix = projectionMatrix(cameraFieldOfView, 1.0, lightNear, lightFar)
        # self.lightProjectionMatrix = orthographicMatrix(-5.0, 5.0, -5.0, 5.0, 1.0, 10.0)

        self.createQuad()
        self.createGroundPlane()
        self.createAxes()
        self.resize()
        

    def resize(self):
        viewport = glGetIntegerv(GL_VIEWPORT)
        width = viewport[2] - viewport[0]
        height = viewport[3] - viewport[1]
        aspect = float(width) / float(height)
        self.projectionMatrix = projectionMatrix(cameraFieldOfView, aspect, cameraNear, cameraFar)


    def lookAt(self, eye, target, up):
        self.viewMatrix = lookAtMatrix(eye, target, up)
        self.viewPos = eye
    
    def unproject(self, winX, winY, z):
              
        viewport = glGetIntegerv(GL_VIEWPORT)
        normX = (winX - viewport[0]) / viewport[2] * 2.0 - 1.0
        normY = (viewport[1] - winY) / viewport[3] * 2.0 - 1.0
        normZ = 2.0 * z - 1.0

        A = np.linalg.inv(self.projectionMatrix.T * self.viewMatrix.T)
        b = A * np.array([normX, normY, normZ, 1.0], dtype=np.float32)

        if b[3] == 0.0:
            return np.zeros(3, dtype=np.float32)
        else:
            return  b[:3] / b[3]
        
    
    def getMouseRay(self, x, y):

        p0 = self.unproject(x, y, 0.0)
        p1 = self.unproject(x, y, 1.0)
        orig = np.array([p0[0], p0[1], p0[2]])
        dir = np.array([p1[0], p1[1], p1[2]]) - orig
        dir /= np.linalg.norm(dir)
        return [orig, dir]


    def setShaderMeshVariables(self, shader, mesh = None):

        if mesh is None:
            loc = glGetUniformLocation(shader, "modelMat")
            if loc >= 0:
                glUniformMatrix4fv(loc, 1, GL_FALSE, np.identity(4, dtype=np.float32))
            loc = glGetUniformLocation(shader, "normalMat")
            if loc >= 0:
                glUniformMatrix3fv(loc, 1, GL_FALSE, np.identity(3, dtype=np.float32))
            return
              
        loc = glGetUniformLocation(shader, "modelMat")
        if loc >= 0:
            glUniformMatrix4fv(loc, 1, GL_FALSE, mesh.transform)

        loc = glGetUniformLocation(shader, "normalMat")
        if loc >= 0:
            normalMatrix = np.linalg.inv(mesh.transform).T
            glUniformMatrix3fv(loc, 1, GL_FALSE, normalMatrix[:3, :3])

        loc = glGetUniformLocation(shader, "hasTexture")
        if loc >= 0:
            glUniform1i(loc, mesh.textureId) 

        loc = glGetUniformLocation(shader, "materialColor")
        if loc >= 0:
            glUniform3f(loc, mesh.color[0], mesh.color[1], mesh.color[2])


    def setShaderVariables(self, shader):

        loc = glGetUniformLocation(shader, "projectionMat")
        glUniformMatrix4fv(loc, 1, GL_TRUE, self.projectionMatrix)

        loc = glGetUniformLocation(shader, "viewMat")
        glUniformMatrix4fv(loc, 1, GL_TRUE, self.viewMatrix)

        loc = glGetUniformLocation(shader, "viewPos")
        if loc >= 0:
            pos = self.viewPos
            glUniform3f(loc, pos[0], pos[1], pos[2])

        numLightsToUse = min(len(self.lights), 4)

        loc = glGetUniformLocation(shader, "numLights")
        if loc >= 0:
            glUniform1i(loc, numLightsToUse)

        loc = glGetUniformLocation(shader, "renderShadows")
        if loc >= 0:
            glUniform1i(loc, renderShadows)

        loc = glGetUniformLocation(shader, "renderLighting")
        if loc >= 0:
            glUniform1i(loc, renderLighting)

        for i in range(numLightsToUse):
            light = self.lights[i]
            loc = glGetUniformLocation(shader, "lights[" + str(i) + "].position")
            if loc >= 0:
                glUniform3f(loc, light.pos[0], light.pos[1], light.pos[2])

            loc = glGetUniformLocation(shader, "lights[" + str(i) + "].range")
            if loc >= 0:
                glUniform1f(loc, light.range)

            loc = glGetUniformLocation(shader, "lights[" + str(i) + "].color")
            if loc >= 0:
                glUniform3f(loc, light.color[0], light.color[1], light.color[2])

            loc = glGetUniformLocation(shader, "lightProjectionMat[" + str(i) + "]")
            if loc >= 0:
                glUniformMatrix4fv(loc, 1, GL_TRUE, self.lightProjectionMatrix)

            loc = glGetUniformLocation(shader, "lightViewMat[" + str(i) + "]")
            if loc >= 0:
                glUniformMatrix4fv(loc, 1, GL_TRUE, light.viewMatrix)

        loc = glGetUniformLocation(shader, "pointScale")
        if loc >= 0:
            viewport = glGetIntegerv(GL_VIEWPORT)
            windowH = viewport[3]
            pointScale = windowH / math.tan(cameraFieldOfView * 0.5 * math.pi / 180.0)
            glUniform1f(loc, pointScale)


    def addMesh(self, mesh):
        self.meshes.append(mesh)


    def addParticles(self, particles):
        self.particles.append(particles)


    def createDefaultLights(self, distance = 5.0):

        light = Light(pos=[distance, distance, distance], at=[0.0, 0.0, 0.0], range=2.0 * distance)
        self.lights.append(light)

        light = Light(pos=[-distance, distance, distance], at=[0.0, 0.0, 0.0], range=2.0 * distance)
        self.lights.append(light)

    def createQuad(self):
        # Vertices of a full screen quad (positions and texture coordinates)
        quadVertices = [0.0, 1.0, 0.0, 1.0,
                        0.0, 0.0, 0.0, 0.0,
                        1.0, 0.0, 1.0, 0.0,
                        1.0, 1.0, 1.0, 1.0]

        self.quadVAO = glGenVertexArrays(1)
        self.quadVBO = glGenBuffers(1)
        glBindVertexArray(self.quadVAO)
        glBindBuffer(GL_ARRAY_BUFFER, self.quadVBO)
        glBufferData(GL_ARRAY_BUFFER, (GLfloat * len(quadVertices))(*quadVertices), GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), None)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), ctypes.c_void_p(2 * sizeof(GLfloat)))
        glBindVertexArray(0)


    def createGroundPlane(self):
        groundNumTiles = 30
        groundTileSize = 0.5

        self.groundVerts = np.zeros(3 * 4 * groundNumTiles * groundNumTiles, dtype = float)
        self.groundColors = np.zeros(3 * 4 * groundNumTiles * groundNumTiles, dtype = float)

        squareVerts = [[0,0], [0,1], [1,1], [1,0]]
        r = groundNumTiles / 2.0 * groundTileSize

        for xi in range(groundNumTiles):
            for zi in range(groundNumTiles):
                x = (-groundNumTiles / 2.0 + xi) * groundTileSize
                z = (-groundNumTiles / 2.0 + zi) * groundTileSize
                p = xi * groundNumTiles + zi
                for i in range(4):
                    q = 4 * p + i
                    px = x + squareVerts[i][0] * groundTileSize
                    pz = z + squareVerts[i][1] * groundTileSize
                    self.groundVerts[3 * q] = px
                    self.groundVerts[3 * q + 2] = pz
                    col = 0.4
                    if (xi + zi) % 2 == 1:
                        col = 0.8
                    pr = math.sqrt(px * px + pz * pz)
                    d = max(0.0, 1.0 - pr / r)
                    col = col * d
                    for j in range(3):
                        self.groundColors[3 * q + j] = col


    def createAxes(self):

        self.axesArrayObject = glGenVertexArrays(1)
        glBindVertexArray(self.axesArrayObject)

        verts = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        
        vertexBuffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer)
        glBufferData(GL_ARRAY_BUFFER, verts, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        colors = np.array([
            [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)

        colorBuffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, colorBuffer)
        glBufferData(GL_ARRAY_BUFFER, colors, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)

        glBindVertexArray(0)


    def createSpriteShader(self):
        self.spriteShader = shaders.compileProgram(
            shaders.compileShader("""
                                  
                #version 330 core
                layout (location = 0) in vec3 aPos;
                layout (location = 1) in vec4 aQuats;
                layout (location = 2) in float aRadii;
                                
                uniform mat4 projectionMat;
                uniform mat4 viewMat;
                uniform vec3 viewPos;
                uniform int useOrientation;
                                  
                uniform float pointScale;   // scale to calculate size in pixels
                uniform float pointRadius;  // point size in world space

                out vec3 pos;
                out vec4 quat;
                out float radius;

                void main()
                {
                    gl_Position = projectionMat * viewMat * vec4(aPos, 1.0);
                                  
                    pos = aPos;
                    radius = pointRadius;
                    if (radius == 0.0)
                        radius = aRadii;
                    if (useOrientation == 1)
                        quat = aQuats;
                    float dist = length(viewPos - pos);
                    gl_PointSize = radius * (pointScale / dist);
                                  
                }
            """, GL_VERTEX_SHADER),

            shaders.compileShader("""
                
                #version 330 core

                const float PI = 3.141592653;
                                  
                int flatShading = 0;
                uniform int useOrientation = 0;
                vec3 lightPos = vec3(3.0, 3.0, 3.0);
                                  
                uniform vec3 viewPos;

                in vec3 pos;
                in vec4 quat;
                in float radius;

                out vec4 fragColor;

                vec3 qtransform(vec4 q, vec3 v)
                {
	                return v + 2.0*cross(cross(v, q.xyz ) + q.w*v, q.xyz);
	            } 
                                  
                void main()
                {
                    if (flatShading == 1)  // for shadow maps
                    { 
                        float x = -1.0 + 2.0 * gl_PointCoord.x;
                        float y = 1.0 - 2.0 * gl_PointCoord.y;
                        if (x * x + y * y > 1.0)
                            discard;
                        else
                            fragColor = vec4(1.0, 1.0, 1.0, 1.0);
                        return;    
                    }
                                                      
                    vec3 lightDir = normalize(lightPos - pos);

                    vec3 a0 = normalize(pos - viewPos);
                    vec3 a2 = vec3(0.0, 1.0, 0.0);
                    vec3 a1 = normalize(cross(a0, a2));
                    a2 = normalize(cross(a1, a0));

                    float x = -radius + 2.0 * gl_PointCoord.x * radius;
                    float y = radius - 2.0 * gl_PointCoord.y * radius;

                    float r2 = x * x + y * y;
                    if (r2 > radius * radius)
                        discard;

                    float h = sqrt(radius * radius - r2);
                    vec3 d = x * a1 + y * a2;
                    vec3 localPos = d - h * a0;
                    vec3 fragPos = pos + localPos;

                    vec3 normal = normalize(localPos);
                                  
                    // to do: provide color as attribute
                    vec4 color = vec4(1.0, 1.0, 1.0, 1.0);

                    if (useOrientation == 1) 
                    {
                        // create a beach ball coloring
                                  
                        vec3 rotNormal = qtransform(quat, normal);

                        vec3 d0 = qtransform(quat, d - h * a0);
                        vec3 d1 = qtransform(quat, d + h * a0);

                        float angle = 0.0;

                        if (d0.z < d1.z) 
                            angle = PI + atan(d0.x, d0.y);
                        else 
                            angle = PI + atan(d1.x, d1.y);
                                    
                        angle = atan(rotNormal.y, rotNormal.x);

                        int segment = int(angle / PI * 6.0);
                                  
                        if (segment % 2 == 0)
                            color = vec4(1.0, 0.0, 0.0, 1.0);
                    }
            
                    float diffuse = max(0.0, dot(lightDir, normal));

                    vec3 viewDir = normalize(viewPos - fragPos);
                    vec3 halfwayDir = normalize(lightDir + viewDir);  
                    float specular = pow(max(dot(normal, halfwayDir), 0.0), 32.0);

                    float ambient = 0.2;

                    fragColor = color * (ambient + diffuse) + specular * vec4(1.0, 1.0, 1.0, 1.0);
                }
            """, GL_FRAGMENT_SHADER)        
        )

    def createDefaultShader(self):
        self.defaultShader = shaders.compileProgram(
            shaders.compileShader("""
                #version 330 core
                layout (location = 0) in vec3 aPos;
                layout (location = 1) in vec3 aNormal;
                layout (location = 2) in vec2 aUv;
                                
                uniform mat4 projectionMat;
                uniform mat4 viewMat;
                uniform mat4 modelMat;
                uniform mat3 normalMat;
                                                                    
                #define MAX_POINT_LIGHTS 4  
                uniform mat4 lightProjectionMat[MAX_POINT_LIGHTS];
                uniform mat4 lightViewMat[MAX_POINT_LIGHTS];

                out vec3 normal;
                out vec2 uv;
                out vec3 fragPos;
                out vec4 fragPosLightSpace[MAX_POINT_LIGHTS];
                                  
                void main()
                {
                    normal = normalMat * aNormal;
                    uv = aUv;
                    gl_Position = projectionMat * viewMat * modelMat * vec4(aPos, 1.0);
                    fragPos = vec3(modelMat * vec4(aPos, 1.0));
                    for (int i = 0; i < MAX_POINT_LIGHTS; i++) 
                        fragPosLightSpace[i] = lightProjectionMat[i] * lightViewMat[i] * vec4(fragPos, 1.0);
                }                              
  
            """, GL_VERTEX_SHADER),

            shaders.compileShader("""
                #version 330 core

                #define MAX_POINT_LIGHTS 4  

                uniform vec3 viewPos;
                uniform sampler2D diffuseTexture;
                uniform sampler2D depthMaps[MAX_POINT_LIGHTS];
                                  
                struct Light {    
                    vec3 position;
                    float range;
                    vec3 color;
                };  

                uniform Light lights[MAX_POINT_LIGHTS];
                uniform int numLights;  
                uniform int renderShadows;
                uniform int renderLighting;
                uniform int hasTexture;
                uniform vec3 materialColor;

                in vec3 normal;
                in vec2 uv;
                in vec3 fragPos;
                in vec4 fragPosLightSpace[MAX_POINT_LIGHTS];
                                                                    
                out vec4 fragColor;
                                  
                float calcPointLight(Light light, vec3 normal, vec3 fragPos, vec3 viewPos, float shadow)
                {                 
                    float ambient = 0.1;
                                                                    
                    vec3 lightDir = normalize(light.position - fragPos);
                    vec3 n = normalize(normal);
                    float diffuse = max(dot(lightDir, n), 0.0);

                    vec3 viewDir = normalize(viewPos - fragPos);
                    vec3 halfwayDir = normalize(lightDir + viewDir);  
                    float specular = pow(max(dot(n, halfwayDir), 0.0), 64.0);

                    float lightDist = length(light.position - fragPos);
                    float attenuation = 1.0;
                    if (lightDist > light.range) 
                        attenuation = 1.0 / (1.0 + lightDist - light.range);
                                  
                    return attenuation * (ambient + (1.0 - shadow) * (diffuse + specular));
                }     

                float calcShadow(vec4 fragPosLightSpace, sampler2D depthMap, float bias)
                {
                    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
                    projCoords = projCoords * 0.5 + 0.5;
                    if (projCoords.x < 0.0 || projCoords.x > 1.0 || projCoords.y < 0.0 || projCoords.y > 1.0)
                        return 0.0;

                    float currentDepth = projCoords.z;

                    float shadow = 0.0;
                    vec2 texelSize = 1.0 / textureSize(depthMap, 0);
                    for (int x = -1; x <= 1; ++x) {
                        for (int y = -1; y <= 1; ++y) {
                            float pcfDepth = texture(depthMap, projCoords.xy + vec2(x, y) * texelSize).r;                                   
                            shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;
                        }
                    }
                    shadow /= 9.0;
                    return shadow;
                }                                              

                void main()
                {                                  
                    int numLightsToUse = min(numLights, MAX_POINT_LIGHTS);
                                                                    
                    float shadow = 0.0;
                                  
                    if (renderShadows == 1) {
                        for (int i = 0; i < numLightsToUse; ++i) {
                            float bias = max(0.05 * (1.0 - dot(normal, lights[i].position - fragPos)), 0.005);
                            shadow += calcShadow(fragPosLightSpace[i], depthMaps[i], bias);
                        }
                        shadow /= float(numLightsToUse);
                    }

                    vec3 lightColor = vec3(0.0);

                    if (renderLighting == 1) {
                        for (int i = 0; i < numLightsToUse; ++i) {
                            float lighting = calcPointLight(lights[i], normal, fragPos, viewPos, shadow);
                            lighting *= 1.0 / float(numLightsToUse);
                            lightColor += lighting * lights[i].color ;
                        }
                    } else {
                        lightColor = vec3(1.0, 1.0, 1.0); 
                    }

                    vec3 matColor = materialColor;
                    if (hasTexture != 0)                                
                        matColor = texture(diffuseTexture, uv).rgb;                                  
                    fragColor = vec4(lightColor * matColor, 1.0);
                }
            """, GL_FRAGMENT_SHADER)        
        )

        glUseProgram(self.defaultShader)  

        loc = glGetUniformLocation(self.defaultShader, "diffuseTexture")
        glUniform1i(loc, 0) 
        
        for i in range(4):
            loc = glGetUniformLocation(self.defaultShader, "depthMaps[" + str(i) + "]")
            glUniform1i(loc, i + 1)
        glUseProgram(0)


    def createFlatShader(self):
        self.flatShader = shaders.compileProgram(
            shaders.compileShader("""
                #version 330 core
                layout (location = 0) in vec3 aPos;
                layout (location = 1) in vec3 aColors;

                uniform int useVertexColors;       
                uniform mat4 projectionMat;
                uniform mat4 viewMat;
                uniform mat4 modelMat;

                out vec3 color;

                void main()
                {
                    gl_Position = projectionMat * viewMat * modelMat * vec4(aPos, 1.0);
//                    gl_Position = viewMat * modelMat * vec4(aPos, 1.0);
                    if (useVertexColors == 1)
                        color = aColors;
                    else
                        color = vec3(1.0, 1.0, 1.0);
                    
                }                              
  
            """, GL_VERTEX_SHADER),

            shaders.compileShader("""
                #version 330 core
                                  
                in vec3 color;
                out vec4 FragColor;

                void main()
                {
                    FragColor = vec4(color, 1.0);
                }
            """, GL_FRAGMENT_SHADER)        
        )

    def createDepthDisplayShader(self):
        self.depthDisplayShader = shaders.compileProgram(
            shaders.compileShader("""
                #version 330 core
                layout (location = 0) in vec2 aPos;
                layout (location = 1) in vec2 aTexCoords;

                out vec2 TexCoords;

                void main() {
                    TexCoords = aTexCoords;
                    gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
                }                         
  
            """, GL_VERTEX_SHADER),

            shaders.compileShader("""
                #version 330 core
                out vec4 FragColor;
                
                in vec2 TexCoords;

                uniform sampler2D depthMap;
                uniform float nearPlane;
                uniform float farPlane;

                float LinearizeDepth(float depth)
                {
                    float z = depth * 2.0 - 1.0; // Back to NDC 
                    return (2.0 * nearPlane * farPlane) / (farPlane + nearPlane - z * (farPlane - nearPlane));
                }

                void main()
                {             
                    float depthValue = texture(depthMap, TexCoords).r;
//                  FragColor = vec4(vec3(LinearizeDepth(depthValue) / farPlane), 1.0); // perspective
                    FragColor = vec4(vec3(depthValue), 1.0); 
                } 
            """, GL_FRAGMENT_SHADER)        
        )


    def renderGroundPlane(self):
        glColor3f(1.0, 1.0, 1.0)
        glNormal3f(0.0, 1.0, 0.0)

        numVerts = math.floor(len(self.groundVerts) / 3)

        glVertexPointer(3, GL_FLOAT, 0, self.groundVerts)
        glColorPointer(3, GL_FLOAT, 0, self.groundColors)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glDrawArrays(GL_QUADS, 0, numVerts)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        
    def renderDepthMap(self, depthMapTexture):

        if self.depthDisplayShader is None:
            self.createDepthDisplayShader()

        glUseProgram(self.depthDisplayShader)

        loc = glGetUniformLocation(self.depthDisplayShader, "nearPlane")
        glUniform1f(loc, cameraNear)

        loc = glGetUniformLocation(self.depthDisplayShader, "farPlane")
        glUniform1f(loc, cameraFar)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, depthMapTexture)
        glUniform1i(glGetUniformLocation(self.depthDisplayShader, "depthMap"), 0)
        glBindVertexArray(self.quadVAO)
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4)
        glBindVertexArray(0)
        glUseProgram(0)



    def renderAxes(self):
        
        if self.flatShader is None or self.axesArrayObject is None:
            return

        glUseProgram(self.flatShader)
        self.setShaderVariables(self.flatShader)
        self.setShaderMeshVariables(self.flatShader)

        glUniform1i(glGetUniformLocation(self.flatShader, "useVertexColors"), 1)

        glBindVertexArray(self.axesArrayObject)
        glDrawArrays(GL_LINES, 0, 6)
        glBindVertexArray(0)
        glUseProgram(0)
          

    def renderToDepthMap(self, light):

        if light.depthFBO == 0:
            light.createDepthMap(depthResolution, depthResolution)

        viewport = glGetIntegerv(GL_VIEWPORT)

        glBindFramebuffer(GL_FRAMEBUFFER, light.depthFBO)
        glViewport(0, 0, depthResolution, depthResolution)
        glClear(GL_DEPTH_BUFFER_BIT)

        glUseProgram(self.flatShader)

        glUniform1i(glGetUniformLocation(self.flatShader, "useVertexColors"), 0)

        loc = glGetUniformLocation(self.flatShader, "projectionMat")
        glUniformMatrix4fv(loc, 1, GL_TRUE, self.lightProjectionMatrix)

        loc = glGetUniformLocation(self.flatShader, "viewMat")
        glUniformMatrix4fv(loc, 1, GL_TRUE, light.viewMatrix)  

        for mesh in self.meshes:

            self.setShaderMeshVariables(self.flatShader, mesh)

            glBindVertexArray(mesh.vertexArrayObject)
            glDrawElements(GL_TRIANGLES, len(mesh.triIds), GL_UNSIGNED_INT, mesh.triIds)
            glBindVertexArray(0)

        glUseProgram(self.spriteShader)

        glEnable(GL_POINT_SPRITE)
        glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE)
        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE)

        glUniform1i(glGetUniformLocation(self.spriteShader, "flatShading"), 1)
        glUniform1i(glGetUniformLocation(self.spriteShader, "useOrientation"), 0)
        loc = glGetUniformLocation(self.spriteShader, "projectionMat")
        glUniformMatrix4fv(loc, 1, GL_TRUE, self.lightProjectionMatrix)

        loc = glGetUniformLocation(self.spriteShader, "viewMat")
        glUniformMatrix4fv(loc, 1, GL_TRUE, light.viewMatrix)  

        loc = glGetUniformLocation(self.spriteShader, "viewPos")
        glUniform3f(loc, light.pos[0], light.pos[1], light.pos[2])  

        for particles in self.particles:
            
            glUniform1f(glGetUniformLocation(self.spriteShader, "pointRadius"), particles.radius)
            glBindVertexArray(particles.vertexArrayObject)        
            glDrawArrays(GL_POINTS, 0, len(particles.pos))
            glBindVertexArray(0)

        glUseProgram(0)
        glDisable(GL_POINT_SPRITE)        

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        glViewport(viewport[0], viewport[1], viewport[2], viewport[3]) 

        # glBindFramebuffer(GL_FRAMEBUFFER, light.depthFBO)
        # depth_values = np.zeros((depthResolution, depthResolution), dtype=np.float32)
        # glReadPixels(0, 0, depthResolution, depthResolution, GL_DEPTH_COMPONENT, GL_FLOAT, depth_values)
        # for i in range(depthResolution):
        #     for j in range(depthResolution):
        #         if depth_values[i,j] != 1.0:
        #             print(depth_values[i, j])
        # glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def render(self):

        if self.spriteShader is None:
            self.createSpriteShader()

        if self.defaultShader is None:
            self.createDefaultShader()

        if self.flatShader is None:
            self.createFlatShader()

        numLightsToUse = min(len(self.lights), 4)

        if renderShadows:

            wasCullFaceEnabled = glIsEnabled(GL_CULL_FACE)
            glEnable(GL_CULL_FACE)
            glCullFace(GL_BACK)

            for i in range(numLightsToUse):
                self.renderToDepthMap(self.lights[i])

                glActiveTexture(GL_TEXTURE0 + i + 1)
                glBindTexture(GL_TEXTURE_2D, self.lights[i].depthTexture)

            if not wasCullFaceEnabled:
                glDisable(GL_CULL_FACE)

            
            glActiveTexture(GL_TEXTURE0)

        glClearColor(.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if renderAxes:
            self.renderAxes()

        if renderDepthMap and renderShadows:
            self.renderDepthMap(self.lights[0].depthTexture)

        if self.defaultShader:

            glUseProgram(self.defaultShader)

            self.setShaderVariables(self.defaultShader)

            # if renderGroundPlane:
            #     self.renderGroundPlane()

            for mesh in self.meshes:

                self.setShaderMeshVariables(self.defaultShader, mesh)

                if mesh.textureId > 0:
                    glActiveTexture(GL_TEXTURE0)
                    glBindTexture(GL_TEXTURE_2D, mesh.textureId)
                
                glBindVertexArray(mesh.vertexArrayObject)

                glDrawElements(GL_TRIANGLES, len(mesh.triIds), GL_UNSIGNED_INT, mesh.triIds)

                glBindVertexArray(0)
                glBindTexture(GL_TEXTURE_2D, 0)

            glUseProgram(0)


        if self.spriteShader:

            glEnable(GL_POINT_SPRITE)
            glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE)
            glEnable(GL_VERTEX_PROGRAM_POINT_SIZE)

            glUseProgram(self.spriteShader)

            glUniform1i(glGetUniformLocation(self.spriteShader, "flatShading"), 0)

            self.setShaderVariables(self.spriteShader)

            for particles in self.particles:

                glUniform1f(glGetUniformLocation(self.spriteShader, "pointRadius"), particles.radius)
                useOrientation = 1 if particles.quats is not None else 0
                glUniform1i(glGetUniformLocation(self.spriteShader, "useOrientation"), useOrientation)

                glBindVertexArray(particles.vertexArrayObject)        
                glDrawArrays(GL_POINTS, 0, len(particles.pos))
                glBindVertexArray(0)

            glDisable(GL_POINT_SPRITE)
            glUseProgram(0)

        for i in range(numLightsToUse):
            glActiveTexture(GL_TEXTURE0 + i + 1)
            glBindTexture(GL_TEXTURE_2D, 0)

        glActiveTexture(GL_TEXTURE0)

        glutSwapBuffers()


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

class Camera:
    
    def __init__(self):
        self.pos = np.array([0.0, 0.5, 2.0])
        self.forward = np.array([0.0, 0.0, -1.0])
        self.up = np.array([0.0, 1.0, 0.0])
        self.right = np.cross(self.forward, self.up)
        self.speed = 0.1
        self.keyDown = [False] * 256


    def rot(self, unitAxis, angle, v):
        
        vrot = v * math.cos(angle) + \
            np.cross(unitAxis, v) * math.sin(angle) + \
            unitAxis * np.dot(unitAxis, v) * (1.0 - math.cos(angle))
        
        return vrot


    def lookAt(self, pos, at):

        self.pos = pos
        self.forward = at - pos
        self.forward = normalize(self.forward)
        self.up = np.array([0.0, 1.0, 0.0])
        self.right = np.cross(self.forward, self.up)
        self.right = normalize(self.right)
        self.up = np.cross(self.right, self.forward)


    def handleMouseTranslate(self, dx, dy):
        
        scale = np.linalg.norm(self.pos) * 0.001
        self.pos = self.pos - self.right * (scale * float(dx))
        self.pos = self.pos + self.up * (scale * float(dy))

    def handleWheel(self, direction):
        
        self.pos = self.pos + self.forward * (direction * self.speed)


    def handleMouseView(self, dx, dy):
        
        scale = 0.005
        self.forward = self.rot(self.up, -dx * scale, self.forward)
        self.forward = self.rot(self.right, -dy * scale, self.forward)
        self.forward = normalize(self.forward)
        self.right = np.cross(self.forward, self.up)
        self.right = np.array([self.right[0], 0.0, self.right[2]])
        self.right = normalize(self.right)
        self.up = np.cross(self.right, self.forward)
        self.up = normalize(self.up)
        self.forward = np.cross(self.up, self.right)
    
    
    def handleKeyDown(self, key):
        
        self.keyDown[ord(key)] = True


    def handleKeyUp(self, key):
        
        self.keyDown[ord(key)] = False


    def handleKeys(self):
        
        if self.keyDown[ord('+')]:
            self.speed = self.speed * 1.2
        if self.keyDown[ord('-')]:
            self.speed = self.speed * 0.8
        if self.keyDown[ord('w')]:
            self.pos = self.pos + self.forward * self.speed
        if self.keyDown[ord('s')]:
            self.pos = self.pos - self.forward * self.speed
        if self.keyDown[ord('a')]:
            self.pos = self.pos - self.right * self.speed
        if self.keyDown[ord('d')]:
            self.pos = self.pos + self.right * self.speed
        if self.keyDown[ord('e')]:
            self.pos = self.pos - self.up * self.speed
        if self.keyDown[ord('q')]:
            self.pos = self.pos + self.up * self.speed


    def handleMouseOrbit(self, dx, dy, center):

        offset = self.pos - center
        offset = [
            np.dot(self.right, offset),
            np.dot(self.forward, offset),
            np.dot(self.up, offset)]

        scale = 0.01
        self.forward = self.rot(self.up, -dx * scale, self.forward)
        self.forward = self.rot(self.right, -dy * scale, self.forward)
        self.up = self.rot(self.up, -dx * scale, self.up)
        self.up = self.rot(self.right, -dy * scale, self.up)

        self.right = np.cross(self.forward, self.up)
        self.right = np.array([self.right[0], 0.0, self.right[2]])
        self.right = normalize(self.right)
        self.up = np.cross(self.right, self.forward)
        self.up = normalize(self.up)
        self.forward = np.cross(self.up, self.right)
        self.pos = center + self.right * offset[0]
        self.pos = self.pos + self.forward * offset[1]
        self.pos = self.pos + self.up * offset[2]


def getMouseRay(x, y):
    viewport = glGetIntegerv(GL_VIEWPORT)
    modelMatrix = glGetDoublev(GL_MODELVIEW_MATRIX)
    projMatrix = glGetDoublev(GL_PROJECTION_MATRIX)

    y = viewport[3] - y - 1
    p0 = gluUnProject(x, y, 0.0, modelMatrix, projMatrix, viewport)
    p1 = gluUnProject(x, y, 1.0, modelMatrix, projMatrix, viewport)
    orig = np.array([p0[0], p0[1], p0[2]])
    dir = np.array([p1[0], p1[1], p1[2]]) - orig
    dir = normalize(dir)
    return [orig, dir]


def glutMouseButtonCallback(button, state, x, y):

    global mouseX
    global mouseY
    global mouseButton
    global shiftDown
    global paused

    mouseX = x
    mouseY = y
    if state == GLUT_DOWN:
        mouseButton = button
    else:
        mouseButton = 0
    shiftDown = glutGetModifiers() & GLUT_ACTIVE_SHIFT
#    ray = renderer.getMouseRay(x, y)
    ray = getMouseRay(x, y)
    
    if shiftDown:        
        if state == GLUT_DOWN:
            if userApp and hasattr(userApp, "onMouseButton"):
                if userApp.onMouseButton(True, ray[0], ray[1]):
                    paused = False

    if state == GLUT_UP:
        if userApp and hasattr(userApp, "onMouseButton"):
            userApp.onMouseButton(False, ray[0], ray[1])


def glutMouseMotionCallback(x, y):
    
    global mouseX
    global mouseY
    global mouseButton
    
    dx = x - mouseX
    dy = y - mouseY
    
    if shiftDown:
        # ray = renderer.getMouseRay(x, y)
        ray = getMouseRay(x, y)
        if userApp and hasattr(userApp, "onMouseMotion"):
            userApp.onMouseMotion(ray[0], ray[1])
    else:
        if mouseButton == GLUT_MIDDLE_BUTTON:
            camera.handleMouseTranslate(dx, dy)
        elif mouseButton == GLUT_LEFT_BUTTON:
            camera.handleMouseView(dx, dy)
        elif mouseButton == GLUT_RIGHT_BUTTON:
            camera.handleMouseOrbit(dx, dy, np.array([0.0, 1.0, 0.0]))

    mouseX = x
    mouseY = y        


def glutMouseWheelCallback(wheel, direction, x, y):
    
    camera.handleWheel(direction)


def glutHandleKeyDown(key, x, y):
    camera.handleKeyDown(key)

    global paused
    global singleStep
    global renderGroundPlane

    if key == b'p':
        paused = not paused
    if key == b'g':
        renderGroundPlane = not renderGroundPlane
    if key == b'm':
        singleStep = True
        paused = False
        
    if userApp and hasattr(userApp, "onKey"):
        userApp.onKey(key, True)        
        

def glutHandleKeyUp(key, x, y):
    camera.handleKeyUp(key)
    if userApp and hasattr(userApp, "onKey"):
        userApp.onKey(key, False)
        
    
def glutHandleSpecialKeyDown(key, x, y):
    if userApp and hasattr(userApp, "onSpecialKey"):
        if key in keyStrings:
            userApp.onSpecialKey(keyStrings[key], True)


def glutHandleSpecialKeyUp(key, x, y):
    if userApp and hasattr(userApp, "onSpecialKey"):
        if key in keyStrings:
            userApp.onSpecialKey(keyStrings[key], False)
    

def glutDisplayCallback():
    pass


def glutReshapeCallback(width, height):
    
    if renderer:
        renderer.resize()


def glutTimerCallback(val):
    
    # show fps in window header
    
    global prevTime
    global frameNr
    global paused
    global singleStep
    
    frameNr = frameNr + 1
    numFpsFrames = 30
    currentTime = time.perf_counter()

    if frameNr % numFpsFrames == 0:
        passedTime = currentTime - prevTime
        prevTime = currentTime
        fps = math.floor(numFpsFrames / passedTime)
        glutSetWindowTitle("OpenGL Viewer " + str(fps) + " fps")

    # callbacks

    if not paused and userApp and hasattr(userApp, "update"):
        userApp.update()
        if singleStep:
            singleStep = False
            paused = True
        

    camera.handleKeys()

    if renderer:
        renderer.lookAt(camera.pos, camera.pos + camera.forward, camera.up)
        renderer.render()

    elapsed_ms = (time.perf_counter() - currentTime) * 1000
    glutTimerFunc(max(0, math.floor((1000.0 / targetFps) - elapsed_ms)), glutTimerCallback, 0)

# -----------------------------------------------------------

def setupOpenGL():
    
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_COLOR_MATERIAL)
    # glEnable(GL_CULL_FACE)

    glEnable(GL_NORMALIZE)
    glEnable(GL_POLYGON_OFFSET_FILL)
    glPolygonOffset(1.0, 1.0)

    global quadric
    quadric = gluNewQuadric()

  

# main

import ctypes


def runApp(app):
    # Load the user32 DLL
    user32 = ctypes.WinDLL('user32')

    # Set the process to be DPI aware
    user32.SetProcessDPIAware()

    global userApp    
    global camera
    global renderer

    camera = Camera()
    userApp = app
    
    glutInit()

    glutInitDisplayMode(GLUT_RGBA)
    glutInitWindowSize(2100, 1500)
    glutInitWindowPosition(10, 10)
    wind = glutCreateWindow("OpenGL Viewer")

    setupOpenGL()

    renderer = Renderer()

    glutDisplayFunc(glutDisplayCallback)
    glutReshapeFunc(glutReshapeCallback)

    glutMouseFunc(glutMouseButtonCallback)
    glutMotionFunc(glutMouseMotionCallback)
    glutMouseWheelFunc(glutMouseWheelCallback)
    glutKeyboardFunc(glutHandleKeyDown)
    glutKeyboardUpFunc(glutHandleKeyUp)
    glutSpecialFunc(glutHandleSpecialKeyDown)
    glutSpecialUpFunc(glutHandleSpecialKeyUp)
    
    glutTimerFunc(math.floor(1000.0 / targetFps), glutTimerCallback, 0)
    
    if userApp and hasattr(userApp, "setup"):
        userApp.setup()

    glutMainLoop()
