from particle import Particle

class PSO:
    def __init__(self, constFunction, x0, bounds, num_particles, maxiter) -> None:
        global num_dimensions
        num_dimensions = len(x0)

        err_best_g = -1
        pos_best_g = []

        swarm =[]
        for i in range(0, num_particles):
            swarm.append(Particle(x0))
        
        i = 0
        while i < maxiter:
            for j in range(0, num_particles):
                swarm[j].fitness(constFunction)
                # check if current position is the best globally
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g = list(swarm[j].position_i)
                    err_best_g = float(swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0, num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_postion(bounds)
            
            i += 1
        