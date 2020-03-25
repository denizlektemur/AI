import random
import numpy as np

class individual:
	def __init__(self, gen):
		self.gen = gen
		self.fitness = 0

	def calculate_fitness(self):
		result_pile_0 = 0
		result_pile_1 = 1

		for i in range(len(self.gen)):
			if (self.gen[i] == 0):
				result_pile_0 += (i+1)
			else:
				result_pile_1 *= (i+1)

		self.fitness = abs(result_pile_0 - 36) + abs(result_pile_1 - 360)
		return self.fitness

	def __lt__(self, other):
		return self.fitness < other.fitness


def random_arr(size):
	temp = []
	for i in range(size):
		temp.append(random.randint(0,1))

	return temp


class population:
	def __init__(self, size):
		self.popu = []
		self.size = size
		for i in range(size):
			person = individual(random_arr(10))
			self.popu.append(person)


	def is_fit(self):
		for i in range(self.size):
			if (self.popu[i].calculate_fitness() == 0):
				return True
		return False

	def breed(self, parent1, parent2):
		gen = [0] * 10

		for i in range(10):
			rng = random.random()
			if rng < 0.5:
				gen[i] = parent1.gen[i]
			else:
				gen[i] = parent2.gen[i]

		return individual(gen)

	def mutate(self, indu):
		indu.gen = random_arr(10)


	def next_gen(self, retain_rate, mutation_rate):
		self.popu.sort()

		children = []

		child_rate = 1 - retain_rate

		amount_of_children =  round(self.size * child_rate)

		for i in range(amount_of_children):
			parents = np.random.choice(np.array(self.popu), size=2, replace=False)
			children.append(self.breed(parents[0], parents[1]))

		for i in range(self.size - amount_of_children):
			children.append(self.popu[i])

		mutation = round(self.size * mutation_rate)

		for i in range(mutation):
			self.mutate(random.choice(children))

		self.popu = children


cards = population(30)
i = 0
while not cards.is_fit():
	i+=1
	print("generation: ", i)
	cards.next_gen(0.5,0.1)

print("generation: ", i, " was succesfull")	