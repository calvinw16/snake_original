import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores, save_every=20, episode_idx=None):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episodes')
    plt.ylabel('Return')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    if episode_idx is not None and episode_idx % save_every == 0:
        plt.savefig("training_rewards.png", dpi=200, bbox_inches="tight")

    plt.show(block=False)
    plt.pause(0.1)
