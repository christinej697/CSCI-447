from particle import Particle

class PSO:
    def __init__(self, x0, num_particles, maxiter, shape1, shape2, num_dimensions, classes, mlp, testoutput) -> None:

        self.err_best_g = 0
        self.pos_best_g = []

        swarm =[]
        for i in range(0, num_particles):
            swarm.append(Particle(x0, shape1, shape2, num_dimensions, classes, mlp, testoutput))
        i = 0
        while i < maxiter:
            for j in range(0, num_particles):
                swarm[j].fitness()
                # check if current position is the best globally
                if swarm[j].f1_i > self.err_best_g :
                    self.pos_best_g = list(swarm[j].position_i)
                    self.err_best_g = float(swarm[j].f1_i)
            # cycle through swarm and update velocities and position
            for j in range(0, num_particles):
                swarm[j].update_velocity(self.pos_best_g)
                swarm[j].update_position()
            i += 1
        