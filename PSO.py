import numpy as np
import random
import matplotlib.pyplot as plt

class Particle:
    def __init__(self,algorithm) -> None:
        self.nvars = algorithm.nvars  # Number of decision variables
        self.position = np.zeros(self.nvars)
        self.velocity = np.zeros(self.nvars)
        self.p_best = np.zeros(self.nvars)
        self.g_best = np.zeros(self.nvars)
        self.p_fitness = 0
        self.lr = algorithm.lr
        self.inertia = algorithm.inertia
        self.g_learning_coef = algorithm.g_learning_coef     # Global learning coefficient
        self.p_learning_coef = algorithm.p_learning_coef     # Personal learning coefficietn
        self.obj_func = algorithm.obj_func # Objective function
        self.var_min = algorithm.var_min # Lower bound of Position
        self.var_max = algorithm.var_max # Upper bound of Position

        self.vel_min = -0.1 * (self.var_max - self.var_min) # Velocity limit of particle
        self.vel_max = -self.vel_min

        # Initailize particle (randomize position)
        for i in range(self.nvars):
            self.position[i] = random.uniform(self.var_min[i],self.var_max[i]) #initialize position
        self.p_fitness = self.fit()
        self.p_best = self.position # Initialize personal best
        
    
    # Calculate fitness of particle
    def fit(self): 
        return self.obj_func(self.position) 

    # Update personal best
    def update_p_best(self):
        fit = self.fit()
        if fit < self.p_fitness:
            self.p_fitness = fit
            self.p_best = self.position


    # Calculate cognitive component
    def cognitive_component(self):
        rand = np.random.rand(self.nvars)
        return self.g_learning_coef * rand * (self.p_best - self.position)
    
    # calculate social component 
    def social_component(self):        
        rand = np.random.rand(self.nvars)
        return self.p_learning_coef * rand * (self.g_best - self.position)

    # Update Velocity of particle
    def update_velocity(self, g_best):
        self.g_best = g_best
        self.velocity = self.inertia * self.velocity + self.cognitive_component() + self.social_component()
    
    # Update particle position
    def move(self):
        self.position = self.position + self.velocity * self.lr

    # Apply position limit
    def limit_position(self):
        for i in range(self.nvars):
            self.position[i] = np.minimum(np.maximum(self.position[i], self.var_min[i]), self.var_max[i])

    # Apply velocity limit
    def limit_velocity(self):
        for i in range(self.nvars):
            self.velocity[i] = np.minimum(np.maximum(self.velocity[i], self.vel_min[i]), self.vel_max[i])

    # Velocity mirror effect
    def velocity_reverse(self):
        for i in range(self.nvars):
            if self.position[i] > self.var_max[i] or self.position[i] < self.var_min[i]:
                self.velocity[i] = -self.velocity[i]


class PSO_Algorithm:
    def __init__(self, obj_func, nvars, lower_bound, upper_bound, inertia_damp_mode=False, plot_result=False) -> None:
        self.population = 500 # Particles number
        self.birdstep = 500   # Maximum number of iteration
        self.nvars = nvars      # Number of decision variables

        self.lr = 0.1
        self.var_min = lower_bound   # Lower Bound of Variables
        self.var_max = upper_bound   # Upper bound of Variables

        self.inertia = 1           # Inertia weight
        self.inertia_damp = 0.99    # Inertia weight damping ratio
        self.g_learning_coef = 1.2        # Global learning coefficient
        self.p_learning_coef = 1.5        # Personal learning coefficient
        self.inertia_damp_mode = inertia_damp_mode # Turn on/off the adaptive inertia damping ratio

        self.g_best = np.zeros((1, self.nvars))  # Global best
        self.p_best = np.zeros((self.population, self.nvars))  # Personal best

        self.p_fitness = np.zeros(self.population)   # Personal fitness
        self.g_fitness = np.inf                         # Global fitness
        self.obj_func = obj_func

        self.particles = []  # Generate Particles
        self.g_fitness_log = []   # Fitness value log
        self.plot_result = plot_result     # plot result if True

    def init_particles(self):
        for i in range(self.population):
            self.particles.append(
                Particle(self)) # pass the object PSO_Algorithm itself to Particle
                #Particle(self.obj_func, self.nvars, self.inertia, self.g_learning_coef, self.p_learning_coef, self.lr, self.var_max, self.var_min))
            if self.particles[i].p_fitness < self.g_fitness:
                self.g_fitness = self.particles[i].p_fitness
                self.g_best = self.particles[i].p_best
    
    # method for setring attribute 
    def set_inertia_damping_mode(self, mode=True):
        self.inertia_damp_mode = mode
    
    def set_plot_mode(self,mode=True):
        self.plot_result = mode

    def set_learning_rate(self, lr):
        self.lr = lr

    def set_population(self, population):
        self.population = population

    def set_max_iteration(self, max_iter):
        self.birdstep = max_iter

    def set_inertia_damp(self, inertia_damp):
        self.inertia_damp = inertia_damp

    def set_g_learning_coef(self, g_learning_coef):
        self.g_learning_coef = g_learning_coef
    
    def set_p_learning_coef(self, p_learning_coef):
        self.p_learning_coef = p_learning_coef

    def move_particles(self):

        for particle in self.particles:
            # Update Velocity
            particle.update_velocity(self.g_best)

            # Apply Velocity Limit
            particle.limit_velocity()

            # Update Position
            particle.move()

            # Velocity Mirror Effect
            particle.velocity_reverse()

            # Apply Position Limit
            particle.limit_position()

            # Evaluation and update Personal Best
            particle.update_p_best()

            # Update Global Best
            if particle.p_fitness < self.g_fitness:
                self.g_fitness = particle.p_fitness
                self.g_best = particle.p_best
    

    def update(self):
        for step in range(self.birdstep):
            self.move_particles()
            self.g_fitness_log.append(self.g_fitness)
            
            print(f'Iteration: {step}   fitness { self.g_fitness}')
            if self.inertia_damp_mode:
                self.inertia = self.inertia * self.inertia_damp

        if self.plot_result:
            self.plot_loss()

    def plot_loss(self):
        
        index = np.array(range(self.birdstep))
        plt.plot(index, self.g_fitness_log)
        plt.title(f'final fitness:{str(self.g_fitness)}')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()
#%%
def objective_func(args):
    """
    Objective: minimize loss = x^2 + y^2 + z^2
    """
    loss = 0
    for i in range(len(args)):
        loss += args[i]**2
    return loss

#%%
def main():

    # Kp1,Kp2,Kp3
    
    lower_bound = np.array([1,1,1])
    upper_bound = np.array([100, 100, 100])
    nvars = 3 # Number of decision variable

    pso = PSO_Algorithm(objective_func, nvars, lower_bound, upper_bound)

    pso.set_inertia_damping_mode()  # inertia damping = True
    pso.set_plot_mode()             # plot = True

    pso.init_particles()            # Initialize particle positino
    pso.update()                    # Start algorithm
    print(f'Best Position: {pso.g_best}')

if __name__ == "__main__":
    main()
    

    





    

        

    
        

# %%
