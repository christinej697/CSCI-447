from particle import Particle

class PSO:
    def __init__(self, x0, num_particles, maxiter, num_dimensions, classes, mlp, testoutput, version, shape1, shape2 = None, shape3 = None, verbose = None) -> None:

        self.err_best_g = 0
        self.loss_best = None
        self.pos_best_g = []

        swarm =[]
        for i in range(0, num_particles):
            if shape2 == None and shape3 == None:
                swarm.append(Particle(x0, num_dimensions, classes, mlp, testoutput, version, shape1))
            elif shape2 != None and shape3 == None:
                swarm.append(Particle(x0, num_dimensions, classes, mlp, testoutput, version, shape1, shape2))
            elif shape2 != None and shape3 != None:
                swarm.append(Particle(x0, num_dimensions, classes, mlp, testoutput, version, shape1, shape2, shape3))

        i = 0
        while i < maxiter:
            for j in range(0, num_particles):
                swarm[j].fitness()
                # check if current position is the best globally
                if swarm[j].perform_i > self.err_best_g :
                    self.pos_best_g = list(swarm[j].position_i)
                    self.err_best_g = float(swarm[j].perform_i)
                    self.loss_best = swarm[j].loss_i
                if verbose:
                    print("The global best for this iteration: ", self.err_best_g)
            # cycle through swarm and update velocities and position
            for j in range(0, num_particles):
                swarm[j].update_velocity(self.pos_best_g)
                swarm[j].update_position()
            i += 1
        