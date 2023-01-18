import random
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torchvision.transforms import transforms


class ClassificationVisualizer:

    def __init__(self, title):

        # Initial parameters
        self.handles = {}  # dictionary of handles per layer
        self.title = title
        self.tensor_to_pil_image = transforms.ToPILImage()

    def draw(self, inputs, labels, outputs):

        # Setup figure
        self.figure = plt.figure(self.title)
        plt.axis('off')
        self.figure.canvas.manager.set_window_title(self.title)
        self.figure.set_size_inches(8, 6)
        plt.suptitle(self.title)
        plt.legend(loc='best')

        inputs = inputs
        batch_size, _, _, _ = list(inputs.shape)

        # print(batch_size)
        # print('1')
        # print(range(batch_size))

        output_probabilities = F.softmax(outputs, dim=1).tolist()
        output_probabilities_apple = [x[0] for x in output_probabilities]
        output_probabilities_ball = [x[1] for x in output_probabilities]
        output_probabilities_banana = [x[2] for x in output_probabilities]
        output_probabilities_bell_pepper = [x[3] for x in output_probabilities]
        output_probabilities_binder = [x[4] for x in output_probabilities]
        output_probabilities_bowl = [x[5] for x in output_probabilities]
        output_probabilities_calculator = [x[6] for x in output_probabilities]
        output_probabilities_camera = [x[7] for x in output_probabilities]
        output_probabilities_cap = [x[8] for x in output_probabilities]
        output_probabilities_cell_phone = [x[9] for x in output_probabilities]
        output_probabilities_cereal_box = [x[10] for x in output_probabilities]
        output_probabilities_coffee_mug = [x[11] for x in output_probabilities]
        output_probabilities_comb = [x[12] for x in output_probabilities]
        output_probabilities_dry_battery = [x[13] for x in output_probabilities]
        output_probabilities_flashlight = [x[14] for x in output_probabilities]
        output_probabilities_food_bag = [x[15] for x in output_probabilities]
        output_probabilities_food_box = [x[16] for x in output_probabilities]
        output_probabilities_food_can = [x[17] for x in output_probabilities]
        output_probabilities_food_cup = [x[18] for x in output_probabilities]
        output_probabilities_food_jar = [x[19] for x in output_probabilities]
        output_probabilities_garlic = [x[20] for x in output_probabilities]
        output_probabilities_glue_stick = [x[21] for x in output_probabilities]
        output_probabilities_greens = [x[22] for x in output_probabilities]
        output_probabilities_hand_towel = [x[23] for x in output_probabilities]
        output_probabilities_instant_noodles = [x[24] for x in output_probabilities]
        output_probabilities_keyboard = [x[25] for x in output_probabilities]
        output_probabilities_kleenex = [x[26] for x in output_probabilities]
        output_probabilities_lemon = [x[27] for x in output_probabilities]
        output_probabilities_lightbulb = [x[28] for x in output_probabilities]
        output_probabilities_lime = [x[29] for x in output_probabilities]
        output_probabilities_marker = [x[30] for x in output_probabilities]
        output_probabilities_mushroom = [x[31] for x in output_probabilities]
        output_probabilities_notebook = [x[32] for x in output_probabilities]
        output_probabilities_onion = [x[33] for x in output_probabilities]
        output_probabilities_orange = [x[34] for x in output_probabilities]
        output_probabilities_peach = [x[35] for x in output_probabilities]
        output_probabilities_pear = [x[36] for x in output_probabilities]
        output_probabilities_pitcher = [x[37] for x in output_probabilities]
        output_probabilities_plate = [x[38] for x in output_probabilities]
        output_probabilities_pliers = [x[39] for x in output_probabilities]
        output_probabilities_potato = [x[40] for x in output_probabilities]
        output_probabilities_rubber_eraser = [x[41] for x in output_probabilities]
        output_probabilities_scissors = [x[42] for x in output_probabilities]
        output_probabilities_shampoo = [x[43] for x in output_probabilities]
        output_probabilities_soda_can = [x[44] for x in output_probabilities]
        output_probabilities_sponge = [x[45] for x in output_probabilities]
        output_probabilities_stapler = [x[46] for x in output_probabilities]
        output_probabilities_tomato = [x[47] for x in output_probabilities]
        output_probabilities_toothbrush = [x[48] for x in output_probabilities]
        output_probabilities_toothpaste = [x[49] for x in output_probabilities]
        output_probabilities_water_bottle = [x[50] for x in output_probabilities]

        random_idxs = random.sample(list(range(batch_size)), k=5 * 5)
        

        for plot_idx, image_idx in enumerate(random_idxs, start=1):

            label = labels[image_idx]
            output_probability_apple = output_probabilities_apple[image_idx]
            output_probability_ball = output_probabilities_ball[image_idx]
            output_probability_banana = output_probabilities_banana[image_idx]
            output_probability_bell_pepper = output_probabilities_bell_pepper[image_idx]
            output_probability_binder = output_probabilities_binder[image_idx]
            output_probability_bowl = output_probabilities_bowl[image_idx]
            output_probability_calculator = output_probabilities_calculator[image_idx]
            output_probability_camera = output_probabilities_camera[image_idx]
            output_probability_cap = output_probabilities_cap[image_idx]
            output_probability_cell_phone = output_probabilities_cell_phone[image_idx]
            output_probability_cereal_box = output_probabilities_cereal_box[image_idx]
            output_probability_coffee_mug = output_probabilities_coffee_mug[image_idx]
            output_probability_comb = output_probabilities_comb[image_idx]
            output_probability_dry_battery = output_probabilities_dry_battery[image_idx]
            output_probability_flashlight = output_probabilities_flashlight[image_idx]
            output_probability_food_bag = output_probabilities_food_bag[image_idx]
            output_probability_food_box = output_probabilities_food_box[image_idx]
            output_probability_food_can = output_probabilities_food_can[image_idx]
            output_probability_food_cup = output_probabilities_food_cup[image_idx]
            output_probability_food_jar = output_probabilities_food_jar[image_idx]
            output_probability_garlic = output_probabilities_garlic[image_idx]
            output_probability_glue_stick = output_probabilities_glue_stick[image_idx]
            output_probability_greens = output_probabilities_greens[image_idx]
            output_probability_hand_towel = output_probabilities_hand_towel[image_idx]
            output_probability_instant_noodles = output_probabilities_instant_noodles[image_idx]
            output_probability_keyboard = output_probabilities_keyboard[image_idx]
            output_probability_kleenex = output_probabilities_kleenex[image_idx]
            output_probability_lemon = output_probabilities_lemon[image_idx]
            output_probability_lightbulb = output_probabilities_lightbulb[image_idx]
            output_probability_lime = output_probabilities_lime[image_idx]
            output_probability_marker = output_probabilities_marker[image_idx]
            output_probability_mushroom = output_probabilities_mushroom[image_idx]
            output_probability_notebook = output_probabilities_notebook[image_idx]
            output_probability_onion = output_probabilities_onion[image_idx]
            output_probability_orange = output_probabilities_orange[image_idx]
            output_probability_peach = output_probabilities_peach[image_idx]
            output_probability_pear = output_probabilities_pear[image_idx]
            output_probability_pitcher = output_probabilities_pitcher[image_idx]
            output_probability_plate = output_probabilities_plate[image_idx]
            output_probability_pliers = output_probabilities_pliers[image_idx]
            output_probability_potato = output_probabilities_potato[image_idx]
            output_probability_rubber_eraser = output_probabilities_rubber_eraser[image_idx]
            output_probability_scissors = output_probabilities_scissors[image_idx]
            output_probability_shampoo = output_probabilities_shampoo[image_idx]
            output_probability_soda_can = output_probabilities_soda_can[image_idx]
            output_probability_sponge = output_probabilities_sponge[image_idx]
            output_probability_stapler = output_probabilities_stapler[image_idx]
            output_probability_tomato = output_probabilities_tomato[image_idx]
            output_probability_toothbrush = output_probabilities_toothbrush[image_idx]
            output_probability_toothpaste = output_probabilities_toothpaste[image_idx]
            output_probability_water_bottle = output_probabilities_water_bottle[image_idx]

            output_probability_list = {"apple":output_probability_apple, "ball":output_probability_ball, "banana":output_probability_banana, "bell pepper":output_probability_bell_pepper, "binder":output_probability_binder, "bowl":output_probability_bowl, "calculator":output_probability_calculator, "camera":output_probability_camera, "cap":output_probability_cap, "cell phone":output_probability_cell_phone, "cereal box":output_probability_cereal_box, "coffee mug":output_probability_coffee_mug, "comb":output_probability_comb, "dry battery":output_probability_dry_battery, "flashlight":output_probability_flashlight, "food bag":output_probability_food_bag, "food box":output_probability_food_box, "food can":output_probability_food_can, "food cup":output_probability_food_cup, "food jar":output_probability_food_jar, "garlic":output_probability_garlic, "glue stick":output_probability_glue_stick, "greens":output_probability_greens, "hand towel":output_probability_hand_towel, "instant noodles":output_probability_instant_noodles, "keyboard":output_probability_keyboard, "kleenex":output_probability_kleenex, "lemon":output_probability_lemon, "lightbulb":output_probability_lightbulb, "lime":output_probability_lime, "marker":output_probability_marker, "mushroom":output_probability_mushroom, "notebook":output_probability_notebook, "onion":output_probability_onion, "orange":output_probability_orange, "peach":output_probability_peach, "pear":output_probability_pear, "pitcher":output_probability_pitcher, "plate":output_probability_plate, "pliers":output_probability_pliers, "potato":output_probability_potato, "rubber eraser":output_probability_rubber_eraser, "scissors":output_probability_scissors, "shampoo":output_probability_shampoo, "soda can":output_probability_soda_can, "sponge":output_probability_sponge, "stapler":output_probability_stapler, "tomato":output_probability_tomato, "toothbrush":output_probability_toothbrush, "toothpaste":output_probability_toothpaste, "water bottle":output_probability_water_bottle}

            max_probability = 0
            max_probability_item = 'None'
            max_probability_idx = 0

            print(len(output_probability_list))

            for output_idx, output in enumerate(output_probability_list):
                if output_probability_list[output] > max_probability:
                    max_probability = output_probability_list[output]
                    max_probability_item = output
                    max_probability_idx = output_idx
                    print(output)
                    print(output_idx)

            # is_apple = True if output_probability_apple > 0.5 else False,
            success = True if (label.data.item() == max_probability_idx) else False

            image_t = inputs[image_idx, :, :, :]
            image_pil = self.tensor_to_pil_image(image_t)

            ax = self.figure.add_subplot(5, 5, plot_idx)  # define a 5 x 5 subplot matrix
            plt.imshow(image_pil)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])

            color = 'green' if success else 'red'
            title = max_probability_item
            title += ' ' + str(round(max_probability*100, 1)) + '%'
            ax.set_xlabel(title, color=color)

        plt.draw()
        key = plt.waitforbuttonpress(0.05)
        if not plt.fignum_exists(1):
            print('Terminating')
            exit(0)
