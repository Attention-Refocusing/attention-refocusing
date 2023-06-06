import numpy as np
import random
from lvis_api.lvis.lvis import LVIS
import json
import matplotlib.pyplot as plt
import seaborn as sns

scenarios = ["jobs costume", "animals", "transportation", "food", "currency", "home devices", "people",
             "cleaning tools",
             "maintenance tools", "cooking tools", "furniture", "home accessories", "buildings", "nature", "rooms",
             "sports", "clothes", "personal stuff", "wild life", "marine life", "ceremony", "toys",
             "facial expressions", "geometry shapes", "morals", "facts", "body parts", "plants", "hair style",
             "musical instruments", "fitness", "fantasy objects", "beauty and health", "medical", "military", "signs",
             "autonomous driving", "story telling", "gaming", "movie", "gestures", "logos", "covers", "billboard",
             "fashion", "product design", "art", "interior design", "emotions", "teaching"]


class ScenariosObjsMapping:
    def __init__(self, ann_path, label_space_path):
        # ######################################################################################################
        #                                             LVIS
        # ######################################################################################################
        self.label_space_path = label_space_path
        self.lvis_dataloader = LVIS(annotation_path=ann_path)
        self.lvis_dataloader.load_anns()
        # Filter out the rare objects; that occurs less than n times:
        self._filter_rare_objs(min_acceptable_occur=90)

        # Create the objects names; dict keys are ids (1-->n) and the values are list with obj names:
        self._lvis_create_categories_names_dict()
        self._print_categories_names()

        # Create LVIS Scenario_obj_map:
        self._lvis_create_scenario_obj_map_dict()

        # ######################################################################################################
        #                               UniDet (COCO + Objects365 + OpenImages + Mapillary)
        # ######################################################################################################
        # Create the objects names; dict keys are ids (0-->n-1) and the values are list with obj names:
        self._unidet_create_categories_names_dict()

        # Get overlapped objects between LVIS and UniDet:
        self._lvis_2_unidet_map()

        # Create unidet Scenario_obj_map:
        self._unidet_create_scenario_obj_map_dict()

        self.visualize_scenario_objs_pie()

    def _filter_rare_objs(self, min_acceptable_occur):
        num_obj_100_occur = 0
        self.selected_categories = {}
        for cat_id in self.lvis_dataloader.cats.keys():
            if self.lvis_dataloader.cats[cat_id]['instance_count'] >= min_acceptable_occur:
                num_obj_100_occur += 1
                self.selected_categories[cat_id] = self.lvis_dataloader.cats[cat_id]
                # print(self.lvis_dataloader.cats[cat_id]['synset'], self.lvis_dataloader.cats[cat_id]['synonyms'],
                #      self.lvis_dataloader.cats[cat_id]["name"])  # str, list, str

    def _lvis_create_categories_names_dict(self):
        self.lvis_categories_names = {}
        for cat_id in self.selected_categories.keys():
            obj_names = []
            for obj_name in self.lvis_dataloader.cats[cat_id]['synonyms']:
                obj_names.append(obj_name.replace("(", "").replace(")", "").replace("_", " "))
            obj_names.append(
                self.lvis_dataloader.cats[cat_id]['synset'].split(".")[0].replace("(", "").replace(")", "").replace("_",
                                                                                                                    " "))
            obj_names.append(
                self.lvis_dataloader.cats[cat_id]['name'].split(".")[0].replace("(", "").replace(")", "").replace("_",
                                                                                                                  " "))
            self.lvis_categories_names[cat_id] = {"obj_names": list(set(obj_names)),
                                             "occurrence_count": self.lvis_dataloader.cats[cat_id]['instance_count']}

    def _print_categories_names(self):
        for cat_id in self.lvis_categories_names.keys():
            print(cat_id, "-->", self.lvis_categories_names[cat_id]["obj_names"])
        print(len(self.lvis_categories_names.keys()))

    def _lvis_create_scenario_obj_map_dict(self):
        scenario_obj_map = {}
        scenario_obj_map["jobs costume"] = [315, 1045]
        scenario_obj_map["animals"] = [76, 99, 102, 107, 162, 176, 186, 225, 241, 280, 359, 378, 379, 401, 422, 447,
                                       453, 493, 496, 501, 507, 522, 569, 578, 624, 699, 723, 786, 794, 802, 861, 899,
                                       900, 929, 943, 1202]
        scenario_obj_map["transportation"] = [3, 32, 94, 114, 118, 173, 178, 191, 200, 207, 208, 218, 276, 441, 471,
                                              692, 700, 701, 703, 766, 800, 876, 922, 950, 1011, 1019, 1020, 1056, 1114,
                                              1115, 1123, 1151, 1152, 1178, 1179, 1186]
        scenario_obj_map["food"] = [5, 7, 12, 22, 25, 27, 37, 45, 53, 73, 80, 81, 87, 104, 116, 122, 150, 154, 157, 158,
                                    169, 175, 183, 197, 201, 217, 226, 229, 239, 241, 242, 243, 246, 248, 266, 283, 297,
                                    303, 309, 322, 327, 331, 342, 357, 387, 415, 417, 419, 435, 447, 448, 490, 510, 514,
                                    515, 528, 529, 564, 579, 613, 638, 639, 641, 647, 666, 682, 384, 689, 707, 709, 726,
                                    734, 735, 736, 753, 755, 771, 774, 776, 799, 801, 806, 816, 819, 832, 838, 843, 844,
                                    904, 906, 912, 916, 986, 1009, 1025, 1038, 1044, 1094, 1099, 1106, 1128, 1172, 1175,
                                    1200, 1203]
        scenario_obj_map["currency"] = [288, 697]
        scenario_obj_map["home devices"] = [2, 4, 11, 70, 112, 284, 296, 305, 329, 370, 373, 375, 421, 429, 534, 565,
                                            604, 655, 687, 696, 698, 739, 845, 848, 865, 866, 881, 1021, 1072, 1077,
                                            1083, 1095, 1096, 1141, 1154, 1155]
        scenario_obj_map["people"] = [793]
        scenario_obj_map["cleaning tools"] = [1, 23, 156, 160, 372, 406, 536, 713, 757, 898, 927, 999, 1093, 1098]
        scenario_obj_map["maintenance tools"] = [120, 125, 143, 298, 537, 549, 556, 621, 746, 919, 923, 954, 1061]
        scenario_obj_map["cooking tools"] = [15, 112, 133, 139, 192, 199, 253, 254, 305, 329, 344, 369, 372, 469, 477,
                                             498, 565, 591, 604, 609, 615, 622, 680, 708, 739, 751, 756, 790, 814, 818,
                                             836, 839, 910, 915, 919, 923, 993, 1000, 1021, 1022, 1024, 1068, 1070,
                                             1091, 1095, 1096, 1100, 1117, 1162, 1164, 1180, 1194]
        scenario_obj_map["furniture"] = [19, 77, 90, 181, 182, 232, 285, 346, 358, 361, 367, 390, 395, 464, 548, 558,
                                         738, 982, 1018, 1050]
        scenario_obj_map["home accessories"] = [4, 24, 29, 61, 65, 66, 79, 86, 102, 110, 194, 195, 235, 271, 350, 351,
                                                385, 440, 444, 461, 570, 617, 626, 628, 629, 645, 694, 804, 814, 817,
                                                818, 837, 840, 860, 915, 955, 957, 1002, 1052, 1068, 1108, 1109, 1139,
                                                1185, 1195]
        scenario_obj_map["buildings"] = [272, 445, 512, 795, 811, 960, 1008, 1059, 1074, 1184]
        scenario_obj_map["nature"] = [90, 99, 176, 358, 359, 704, 861, 1064, 1202]
        scenario_obj_map["rooms"] = [68, 77, 261, 430, 442, 609, 679, 795, 955, 957, 961, 1002, 1097, 1136]
        scenario_obj_map["sports"] = [41, 56, 57, 58, 59, 60, 111, 451, 474, 502, 552, 556, 601, 614, 636, 677, 704,
                                      728, 745, 762, 899, 903, 924, 962, 964, 965, 966, 967, 970, 976, 980, 1017, 1037,
                                      1041, 1045, 1078, 1079, 1169, 1177, 1191]
        scenario_obj_map["clothes"] = [15, 34, 35, 36, 48, 59, 75, 88, 92, 95, 111, 115, 132, 137, 138, 148, 152, 203,
                                       274, 277, 278, 319, 392, 394, 459, 500, 544, 547, 550, 589, 592, 595, 636, 644,
                                       695, 716, 749, 828, 870, 911, 921, 947, 948, 951, 953, 968, 973, 981, 995, 1033,
                                       1035, 1036, 1041, 1042, 1043, 1045, 1060, 1122, 1127, 1134, 1142, 1197, 1198]
        scenario_obj_map["personal stuff"] = [1, 34, 35, 36, 54, 66, 72, 108, 133, 146, 189, 198, 212, 230, 252, 259,
                                              271, 404, 409, 524, 536, 586, 605, 630, 631, 841, 885, 995, 1102, 1133,
                                              1156, 1161]
        scenario_obj_map["wild life"] = [29, 44, 76, 162, 358, 359, 493, 496, 578, 653, 861, 1064, 1202]
        scenario_obj_map["marine life"] = [118, 171, 200, 643, 644, 676, 786, 843, 929, 932, 999, 1045, 1177]
        scenario_obj_map["ceremony"] = [28, 83, 84, 104, 137, 138, 152, 157, 183, 194, 197, 255, 344, 347, 377, 461,
                                        494, 498, 629, 655, 685, 713, 753, 756, 771, 832, 1175, 1188, 1190]
        scenario_obj_map["toys"] = [41, 43, 70, 256, 294, 341, 380, 401, 611, 637, 670, 809, 1071, 1110]
        scenario_obj_map["facial expressions"] = []
        scenario_obj_map["geometry shapes"] = [341, 385, 980]
        scenario_obj_map["morals"] = []
        scenario_obj_map["facts"] = []
        scenario_obj_map["body parts"] = [186, 553]
        scenario_obj_map["plants"] = [22, 25, 44, 641, 709, 734, 755, 773, 789, 806, 807, 838, 854, 867, 872, 1025,
                                      1034, 1044, 1099, 1129, 1172, 1203]
        scenario_obj_map["hair style"] = [54]
        scenario_obj_map["musical instruments"] = [521, 655, 685, 797, 798]
        scenario_obj_map["fitness"] = [41, 896, 1041]
        scenario_obj_map["fantasy objects"] = [190, 594]
        scenario_obj_map["beauty and health"] = [54, 534, 675, 744, 935, 979, 1102, 1103, 1104]
        scenario_obj_map["medical"] = [5, 47, 614, 675, 683, 744]
        scenario_obj_map["military"] = [3, 436, 614]
        scenario_obj_map["signs"] = [787, 959, 1019, 1020, 1026, 1027, 1173]
        scenario_obj_map["autonomous driving"] = [3, 114, 173, 178, 191, 207, 208, 218, 276, 298, 441, 627, 642, 668,
                                                  766, 800, 922, 1019, 1020, 1026, 1027, 1056, 1112, 1114, 1115, 1123]
        scenario_obj_map["story telling"] = [294]
        scenario_obj_map["gaming"] = [698, 705, 706]
        scenario_obj_map["movie"] = [880, 1076, 1121, 1143]
        scenario_obj_map["gestures"] = []
        scenario_obj_map["logos"] = [129, 835, 1055]
        scenario_obj_map["covers"] = [129, 658, 834]
        scenario_obj_map["billboard"] = [50, 96, 787, 835, 959]
        scenario_obj_map["fashion"] = [15, 34, 35, 36, 48, 88, 89, 115, 137, 138, 146, 152, 203, 252, 277, 319, 334,
                                       392, 394, 411, 459, 544, 550, 589, 592, 595, 695, 715, 716, 749, 841, 885, 921,
                                       951, 968, 995, 1008, 1033, 1035, 1036, 1042, 1043, 1052, 1122, 1127, 1133, 1142]
        scenario_obj_map["product design"] = [96, 129, 212, 658, 949]
        scenario_obj_map["art"] = [15, 670, 747, 748, 928]
        scenario_obj_map["interior design"] = [2, 19, 24, 68, 235, 312, 346, 350, 351, 361, 367, 609, 617, 626, 628,
                                               738, 928]
        scenario_obj_map["emotions"] = []
        scenario_obj_map["teaching"] = [97, 109, 127, 128, 267, 294, 296, 659, 669, 670, 698, 719, 724, 725, 781, 782]
        scenario_obj_map["others"] = [53, 177, 185, 204, 220, 263, 299, 306, 324, 330, 335, 424, 621, 627, 630, 633,
                                      659, 661, 667, 668, 700, 898, 949, 1066, 1085, 1086]
        self.lvis_scenario_obj_map = scenario_obj_map

    def _unidet_create_categories_names_dict(self):
        self.unidet_categories_names = {}
        unified_label_file = json.load(open(self.label_space_path))
        for cat_id, cat in enumerate(unified_label_file['categories']):
            cat_name = cat['name'].replace("/", "_").lower()
            if cat_name[0] == "_":
                obj_names = cat_name[1:].split('_')
            elif cat_name[-1] == "_":
                obj_names = cat_name[:-1].split('_')
            else:
                obj_names = cat_name.split('_')
            # remove "" from names:
            obj_names = [obj_name for obj_name in obj_names if obj_name != '']
            for i, obj_name in enumerate(obj_names):
                if obj_name != '':
                    obj_names[i] = obj_name.replace("_", " ").lower()
            self.unidet_categories_names[cat_id] = list(set(obj_names))  # remove repeated names

    def _lvis_2_unidet_map(self):
        self.lvis_unidet_map = {}
        self.unidet_lvis_map = {}
        overlaped_obj_count = 0
        for cat_id, obj_names in self.unidet_categories_names.items():
            found_flag = False
            for obj_name in obj_names:
                for lvis_cat_id, lvis_cat in self.lvis_categories_names.items():
                    for lvis_obj_name in lvis_cat['obj_names']:
                        if (obj_name in lvis_obj_name) or (lvis_obj_name in obj_name):
                            found_flag = True
                            overlaped_obj_count += 1
                            self.unidet_lvis_map[cat_id] = lvis_cat_id
                            if lvis_cat_id in self.lvis_unidet_map.keys():
                                self.lvis_unidet_map[lvis_cat_id].append(cat_id)
                            else:
                                self.lvis_unidet_map[lvis_cat_id] = [cat_id]
                            break
                    if found_flag:
                        break
                if found_flag:
                    break
            if not found_flag:
                print(cat_id, "-->", obj_names, "Not found")
        print("overlaped_obj_count: ", overlaped_obj_count)
        print("length of lvis_unidet_map dict:", len(self.lvis_unidet_map))
        print("length of unidet_lvis_map dict:", len(self.unidet_lvis_map))

    def _unidet_create_scenario_obj_map_dict_from_lvis(self):
        self.unidet_scenario_obj_map_from_lvis = {}
        for scenario_name, lvis_objs_id in self.lvis_scenario_obj_map.items():
            objs_id = []
            for lvis_obj_id in lvis_objs_id:
                if lvis_obj_id in self.lvis_unidet_map.keys():
                    objs_id.append(self.lvis_unidet_map[lvis_obj_id])
            self.unidet_scenario_obj_map_from_lvis[scenario_name] = [item for sublist in objs_id for item in sublist]

    def _unidet_create_scenario_obj_map_dict_hardcoded(self):
        scenario_obj_map = {}
        scenario_obj_map["jobs costume"] = [447]
        scenario_obj_map["animals"] = [84, 92, 94, 104, 107, 108, 110, 112, 117, 120, 390, 402, 411, 412, 416, 431, 443,
                                       448, 449, 453, 456, 490, 492, 493, 511, 515, 532, 541, 543, 553, 557, 559, 567,
                                       609, 614, 620, 649, 661, 671, 680, 681, 682]
        scenario_obj_map["transportation"] = [13, 82, 123, 211, 287, 432, 470, 531, 543, 593, 596]
        scenario_obj_map["food"] = [85, 113, 278, 319, 328, 332, 333, 352, 367, 387, 393, 418, 437, 504, 509]
        scenario_obj_map["currency"] = []
        scenario_obj_map["home devices"] = [264, 335, 372, 494, 530]
        scenario_obj_map["people"] = []
        scenario_obj_map["cleaning tools"] = [236, 348, 619]
        scenario_obj_map["maintenance tools"] = [310, 313, 342, 361, 370, 373, 405, 521, 613]
        scenario_obj_map["cooking tools"] = [335, 354, 422, 441, 475, 603, 692, 693]
        scenario_obj_map["furniture"] = [223, 473, 566]
        scenario_obj_map["home accessories"] = [248, 282, 562, 646]
        scenario_obj_map["buildings"] = [233, 386, 406, 421, 481, 508, 519, 525, 542, 568, 610, 635, 638, 695]
        scenario_obj_map["nature"] = [43, 92, 94, 108, 112, 120, 443, 448, 449, 453, 456, 490, 493, 505, 553, 557, 559,
                                      567, 624, 661, 671, 681, 682]
        scenario_obj_map["rooms"] = [259]
        scenario_obj_map["sports"] = [43, 99, 269, 293, 302, 447, 519, 560]
        scenario_obj_map["clothes"] = [400, 419, 548]
        scenario_obj_map["personal stuff"] = [354]
        scenario_obj_map["wild life"] = [385, 94, 412, 416, 431, 448, 490, 493, 511, 515, 567, 609, 661, 671]
        scenario_obj_map["marine life"] = [84, 104, 110, 117, 359, 371, 382, 385, 411, 427, 442, 492, 529, 538, 593,
                                           649]
        scenario_obj_map["ceremony"] = [316, 325, 418, 437]
        scenario_obj_map["toys"] = [287, 294, 590]
        scenario_obj_map["facial expressions"] = []
        scenario_obj_map["geometry shapes"] = []
        scenario_obj_map["morals"] = []
        scenario_obj_map["facts"] = []
        scenario_obj_map["body parts"] = [631]
        scenario_obj_map["plants"] = [278, 328, 358, 362, 363, 380, 505, 580, 624]
        scenario_obj_map["hair style"] = []
        scenario_obj_map["musical instruments"] = [32, 81, 109, 229, 275, 327, 336, 365, 444, 482, 487, 544, 631, 652,
                                                   662]
        scenario_obj_map["fitness"] = [99]
        scenario_obj_map["fantasy objects"] = []
        scenario_obj_map["beauty and health"] = [273, 356, 379, 380, 401, 624]
        scenario_obj_map["medical"] = [123, 479, 485, 575]
        scenario_obj_map["military"] = [48, 82, 111, 455, 464, 465, 558, 598, 605, 630]
        scenario_obj_map["signs"] = [219, 337, 607]
        scenario_obj_map["autonomous driving"] = [13, 123, 596, 607]
        scenario_obj_map["story telling"] = []
        scenario_obj_map["gaming"] = [296]
        scenario_obj_map["movie"] = [296]
        scenario_obj_map["gestures"] = []
        scenario_obj_map["logos"] = []
        scenario_obj_map["covers"] = []
        scenario_obj_map["billboard"] = []
        scenario_obj_map["fashion"] = [400, 401, 419, 548]
        scenario_obj_map["product design"] = []
        scenario_obj_map["art"] = []
        scenario_obj_map["interior design"] = []
        scenario_obj_map["emotions"] = []
        scenario_obj_map["teaching"] = [252, 261, 286, 330, 349, 357, 507]
        scenario_obj_map["others"] = [243, 270, 284, 311, 368, 435]
        self.unidet_scenario_obj_map = scenario_obj_map

    def _fuse_two_dicts_of_list(self, x, y):
        for k, v in x.items():
            if k in y.keys():
                y[k] += v
            else:
                y[k] = v
        return y

    def _unidet_create_scenario_obj_map_dict(self):
        self._unidet_create_scenario_obj_map_dict_from_lvis()
        self._unidet_create_scenario_obj_map_dict_hardcoded()
        # fuse both dicts:
        self.unidet_scenario_obj_map = self._fuse_two_dicts_of_list(self.unidet_scenario_obj_map_from_lvis,
                                                                    self.unidet_scenario_obj_map)

    def visualize_scenario_objs_pie(self):
        labels = list(self.unidet_scenario_obj_map.keys())
        data = [len(v) for k, v in self.unidet_scenario_obj_map.items()]
        # data = [data[i] / sum(data) for i in range(len(data)) if int(100 * data[i] / sum(data)) > 1]
        old_sum = sum(data)
        labels = [labels[i] for i in range(len(data)) if data[i] > 5]
        others = sum([data[i] / sum(data) for i in range(len(data)) if data[i] <= 5])
        data = [data[i] / sum(data) for i in range(len(data)) if data[i] > 5]
        # fuse others
        data[labels.index('others')] += others
        # Shuffle the data:
        temp = list(zip(labels, data))
        random.shuffle(temp)
        labels, data = zip(*temp)
        labels, data = list(labels), list(data)

        # create explode
        exp = np.zeros(len(data))
        exp[labels.index('others')] = 0.1
        pct = '%1.0f%%'
        colors = sns.color_palette('pastel')[0:len(data)]

        plt.figure(figsize=(10, 9), dpi=300)
        plt.pie(data, labels=labels, explode=exp, colors=colors, pctdistance=0.8,
                autopct=pct, startangle=0, rotatelabels=True, normalize=True, shadow=False)
        # plt.title("Objects distribution across scenarios")

        # plt.pie(data, labels=labels, colors=colors, autopct='%.0f%%')
        plt.savefig('visualize_scenario_objs_pie.png', bbox_inches='tight')


if __name__ == "__main__":
    ScenariosObjsMapping(ann_path="../../../data/metrics/det/lvis_v1/lvis_v1_train.json",
                         label_space_path="UniDet-master/datasets/label_spaces/learned_mAP+M.json")
    print("Done !")
