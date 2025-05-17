from humanoid_env import HumanoidEnv
import time
import pybullet as p

if __name__ == "__main__":
    env = HumanoidEnv(render=True)
    obs = env.reset()  # Call reset to initialize the world

    # 1. Check the state of the world after the first reset
    print("Initial world state:")
    print("Number of bodies:", p.getNumBodies())  # Check how many bodies are in the world
    initial_body_ids = [p.getBodyInfo(i)[0] for i in range(p.getNumBodies())] #get the body ids
    print("Body IDs:", initial_body_ids)

    try:
        for _ in range(1000):
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            if done:
                print("Episode done, resetting...")
                obs = env.reset() #call reset again, to check if the world is cleared.

                # 2. Check the state of the world after the reset is called inside the loop.
                print("\nWorld state after reset within the loop:")
                print("Number of bodies:", p.getNumBodies())
                current_body_ids = [p.getBodyInfo(i)[0] for i in range(p.getNumBodies())]
                print("Body IDs:", current_body_ids)

                if p.getNumBodies() == len(initial_body_ids):
                    print("World reset correctly: Number of bodies is the same.")
                else:
                    print("World reset INCORRECTLY: Number of bodies changed!")

                if set(initial_body_ids) == set(current_body_ids):
                    print("World reset correctly: Body IDs are the same.")
                else:
                    print("World reset INCORRECTLY: Body IDs changed!")
                break #remove the time.sleep and break here.

    except KeyboardInterrupt:
        print("Interrupted")

    print("Holding simulation for inspection...")
    time.sleep(20)  # So GUI stays open

    env.close()
