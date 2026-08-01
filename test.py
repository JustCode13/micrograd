from utils.seed import SeedManager

seed_manager = SeedManager(20)
seed_manager.apply()
generator = seed_manager.random_generator()

print(generator.random())
print(generator.random())
print(generator.random())
