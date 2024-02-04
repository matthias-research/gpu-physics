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

import os
import numpy as np
import warp as wp
import viewer

wp.init()

# pip install warp-lang
# pip install numpy

@wp.struct
class ParticleData:
    pos: wp.array(dtype=wp.vec3)
    vel: wp.array(dtype=wp.vec3)
    pos_corr: wp.array(dtype=wp.vec3)
    vel_corr: wp.array(dtype=wp.vec3)    


@wp.struct
class Params:
    dt: float
    num_substeps: int
    jacobi_scale: float
    gravity: wp.vec3
    particle_radius: float
    mesh_max_dist: float


@wp.kernel
def dev_compute_tri_bounds(
        verts: wp.array(dtype=wp.vec3),
        tri_ids: wp.array(dtype=int),
        lowers: wp.array(dtype=wp.vec3),
        uppers: wp.array(dtype=wp.vec3)):
    
    tri_id = wp.tid()

    p0 = verts[tri_ids[3 * tri_id + 0]]
    p1 = verts[tri_ids[3 * tri_id + 1]]
    p2 = verts[tri_ids[3 * tri_id + 2]]
               
    lowers[tri_id] = wp.min(wp.min(p0, p1), p2)
    uppers[tri_id] = wp.max(wp.max(p0, p1), p2)
    

@wp.kernel
def dev_integrate_particles(
        params: Params,
        particles: ParticleData):

    nr = wp.tid()

    dt = params.dt / float(params.num_substeps)

    p = particles.pos[nr]
    v = particles.vel[nr]
    
    v += dt * params.gravity
    p += dt * v
    if p[1] < params.particle_radius:
        p[1] = params.particle_radius
        v[1] = 0.0

    particles.pos[nr] = p
    particles.vel[nr] = v


@wp.func
def dev_handle_particle_collision(
        id0: int, id1: int,
        params: Params,
        particles: ParticleData):

    p0 = particles.pos[id0]
    v0 = particles.vel[id0]

    p1 = particles.pos[id1]
    v1 = particles.vel[id1]

    dp = p1 - p0
    d = wp.length(dp)
    min_dist = params.particle_radius * 2.0

    if d > min_dist:
        return 0
    
    # position correction

    n = wp.normalize(dp)
    corr = n * (min_dist - d) * 0.5

    wp.atomic_add(particles.pos_corr, id0, -corr)
    wp.atomic_add(particles.pos_corr, id1, corr)

    # velocity correction (inelastic)

    vn0 = wp.dot(v0, n)
    vn1 = wp.dot(v1, n)

    avg_v = (vn0 + vn1) * 0.5

    wp.atomic_add(particles.vel_corr, id0, (avg_v - vn0) * n)
    wp.atomic_add(particles.vel_corr, id1, (avg_v - vn1) * n)


@wp.kernel
def dev_handle_particle_collisions(
        grid : wp.uint64,
        params: Params,
        particles: ParticleData):

    id0 = wp.tid()

    p0 = particles.pos[id0]
    r = params.particle_radius

    # find neighbors
    
    query = wp.hash_grid_query(grid, p0, 2.0 * r)
    id1 = int(0)

    while(wp.hash_grid_query_next(query, id1)):

        if id0 < id1:
            dev_handle_particle_collision(id0, id1, params, particles)


@wp.kernel
def dev_apply_corrections(
        params: Params,
        particles: ParticleData):

    nr = wp.tid()

    particles.pos[nr] = particles.pos[nr] + particles.pos_corr[nr] * params.jacobi_scale
    particles.vel[nr] = particles.vel[nr] + particles.vel_corr[nr] * params.jacobi_scale


@wp.kernel
def dev_handle_mesh_collisions(
        mesh_id: wp.uint64,
        params: Params,
        particles: ParticleData):

    nr = wp.tid()
    p = particles.pos[nr]

    query = wp.mesh_query_point(mesh_id, p, params.mesh_max_dist) 

    if  query.result:

        particles.pos[nr] = wp.vec3(0.0, 0.0, 0.0)

        closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)

        n = wp.normalize(p - closest) * query.sign

        dist = wp.dot(n, closest - p) + params.particle_radius

        if dist > 0.0:

            # position correction
            p += n * dist
            particles.pos[nr] = p

            # velocity correction (inelastic, frictionless)
            vn = wp.dot(particles.vel[nr], n)
            particles.vel[nr] -= n * vn


class ParticleSolver:

    def __init__(self, pos, radius, coll_verts=None, coll_tri_ids=None, device = "cuda"):

        self.device = device

        # create particle data

        self.num_particles = len(pos)

        self.particle_data = ParticleData() 
        self.particle_data.pos = wp.array(pos, dtype=wp.vec3, device=device)
        self.particle_data.vel = wp.zeros(shape=(self.num_particles), dtype=wp.vec3, device=device)
   
        self.particle_data.pos_corr = wp.zeros(shape=(self.num_particles), dtype=wp.vec3, device=device)
        self.particle_data.vel_corr = wp.zeros(shape=(self.num_particles), dtype=wp.vec3, device=device)

        self.host_particle_data = ParticleData()
        self.host_particle_data.pos = wp.zeros(shape=(self.num_particles), dtype=wp.vec3, device="cpu")

        # create parameters

        self.params = Params()
        self.params.dt = 0.01
        self.params.num_substeps = 5
        self.params.jacobi_scale = 0.5
        self.params.gravity = wp.vec3(0.0, -10.0, 0.0)   
        self.params.particle_radius = radius
        self.params.mesh_max_dist = 5.0 * radius
        
        # hash for particle collisions

        self.hash_grid = wp.HashGrid(dim_x=128, dim_y=128, dim_z=128, device=device)
        self.hash_radius = 2.0 * radius

        # warp mesh for mesh collisions

        self.coll_mesh = None

        if coll_verts is not None and coll_tri_ids is not None:

            self.coll_verts = wp.array(coll_verts, dtype=wp.vec3, device=device)
            self.coll_tri_ids = wp.array(coll_tri_ids, dtype=int, device=device)
            self.coll_mesh = wp.Mesh(self.coll_verts, self.coll_tri_ids)



    def simulate(self):

        for substep in range(self.params.num_substeps):

            wp.launch(kernel = dev_integrate_particles, 
                inputs = [self.params, self.particle_data],
                dim = self.num_particles, device=self.device) 
                        
            self.hash_grid.build(points=self.particle_data.pos, radius=self.hash_radius)

            self.particle_data.pos_corr.zero_()
            self.particle_data.vel_corr.zero_()

            wp.launch(kernel = dev_handle_particle_collisions, 
                inputs = [self.hash_grid.id, self.params, self.particle_data],
                dim = self.num_particles, device=self.device) 
            
            wp.launch(kernel = dev_apply_corrections, 
                inputs = [self.params, self.particle_data],
                dim = self.num_particles, device=self.device) 
            
            # solve mesh collisions

            if self.coll_mesh is not None:

                wp.launch(kernel = dev_handle_mesh_collisions, 
                    inputs = [self.coll_mesh.id, self.params, self.particle_data],
                    dim = self.num_particles, device=self.device) 
        

    def read_back(self):
                
        wp.copy(self.host_particle_data.pos, self.particle_data.pos)


def create_grid_positions(num, d):
    pos = np.zeros((num * num * num, 3), dtype=np.float32)
    grid = np.mgrid[0:num, 0:num, 0:num].astype(np.float32) * d
    pos[:, :] = grid.reshape(3, -1).T
    return pos


class Example:
             
    def __init__(self, device="cuda"):
        self.device=device
        self.solver = None
        self.viewer_particles = None


    def setup(self):

        mesh = viewer.Mesh()
        dataDir = os.path.dirname(os.path.realpath(__file__)) + os.sep + "data" + os.sep

        mesh.loadObjFile(dataDir + "monkeys.obj", dataDir + "monkeys.png")
        viewer.renderer.addMesh(mesh)

        num = 10
        particle_radius = 0.02
        d = particle_radius * 2.2
        h = 1.0

        pos = create_grid_positions(num, d)
        pos[:, 1] += h

        self.solver = ParticleSolver(
            pos=pos, radius=particle_radius,
            coll_verts=mesh.verts, coll_tri_ids=mesh.triIds,
            device=self.device) 
        
        self.viewer_particles = viewer.Particles(pos=pos, quats=None, radius=particle_radius, radii=None)
        viewer.addParticles(self.viewer_particles)

        size = 2.0

        viewer.camera.lookAt(np.array([0.0, size, 2.0*size]), np.array([0.0, 0.0, 0.0]))
        viewer.camera.speed = 0.2
        viewer.createDefaultLights(5.0 * size)

        viewer.renderShadows = True
        viewer.renderLighting = True


    def update(self):

        self.solver.simulate()
        self.solver.read_back()
        self.viewer_particles.update(
            self.solver.host_particle_data.pos.numpy())
        
        viewer.camera.pos[1] = max(viewer.camera.pos[1], 0.0)


    def render(self, depth=False):
        pass


    def onMouseButton(self, down: bool, orig, dir):
        pass


    def onMouseMotion(self, orig, dir):
        pass        
    
    def onKey(self, key, down):
        pass

            
    def onSpecialKey(self, key, down):
        pass


viewer.runApp(Example())
