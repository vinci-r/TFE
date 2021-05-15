import numpy as np
import sklearn
import sklearn.model_selection
import sklearn.tree
import sklearn.metrics

from math import floor, ceil
from copy import deepcopy

# extract a tree and evaluate its performance
def extract_tree_evaluation(classifier, X_train, X_test, y_train, y_test):
	
	y_pred = classifier.predict(X_train)

	X_new = []
	y_new = []

	# the final tree is built based on well classified examples
	for i in range(y_train.shape[0]):
		if y_pred[i] == y_train[i]:
			X_new.append(X_train[i])
			y_new.append(y_train[i])

	X_new = np.array(X_new)
	y_new = np.array(y_new)


	X_train_old, y_train_old = X_new, y_new

	X_new, X_val, y_new, y_val =\
		   sklearn.model_selection.train_test_split(X_train, y_train, test_size=0.3333333333333, random_state = 42)

	best_acc = -1
	best_conf = 0
	best_depth = -1

	# hyperparameter optimization (tree depth)
	for i in range(10, 70):
		tree_classifier = sklearn.tree.DecisionTreeClassifier(max_depth=i, random_state=42)
		tree_classifier = tree_classifier.fit(X_new, y_new)

		y_pred = tree_classifier.predict(X_val)

		confusion_matrix = sklearn.metrics.confusion_matrix(y_val, y_pred)
		tn, fp, fn, tp = confusion_matrix.ravel()
		accuracy = (tp + tn) / y_val.shape[0]

		if accuracy > best_acc:
			best_acc = accuracy
			best_conf = confusion_matrix
			best_depth = i


	# evaluate model performance

	X_new, y_new = X_train_old, y_train_old

	tree_classifier = sklearn.tree.DecisionTreeClassifier(random_state=42, max_depth=best_depth)
	tree_classifier = tree_classifier.fit(X_new, y_new)


	y_pred = tree_classifier.predict(X_test)

	confusion_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred)
	tn, fp, fn, tp = confusion_matrix.ravel()
	accuracy = (tp + tn) / y_test.shape[0]

	print(best_depth, accuracy)

# simplify the rules with several lower/upper bounds by
# removing redundant constraints
def simplify_rules(rules):
	
	rules2 = deepcopy(rules)
	new_rules = []

	for rule in rules2:
		
		upper_bounds = {}
		lower_bounds = {}
		new_rule = []

		for constraint in rule[0]:
			
			if constraint[1] == "<=":
				if constraint[0] not in upper_bounds:
					upper_bounds[constraint[0]] = constraint[2]
			
				elif constraint[2] < upper_bounds[constraint[0]]:
					upper_bounds[constraint[0]] = constraint[2]

			if constraint[1] == ">":
				if constraint[0] not in lower_bounds:
					lower_bounds[constraint[0]] = constraint[2]
			
				elif constraint[2] > lower_bounds[constraint[0]]:
					lower_bounds[constraint[0]] = constraint[2]

		for feature in lower_bounds:
			new_rule.append([feature, ">", lower_bounds[feature]])

		for feature in upper_bounds:
			new_rule.append([feature, "<=", upper_bounds[feature]])

		new_rules.append([new_rule, rule[1]])

	return new_rules

# only (per prefix) equality constraint is possible, each inequality constraint
#  is to be replaced by as many rules as there are prefixes of interest
# (the function is thus valid only for the prefixes in the given dataset)

def ip_rule_adapter(rules):

	rules2 = deepcopy(rules)
	new_rules = []

	for rule in rules2:
		
		for constraint in rule[0]:
			if constraint[0] == "dst":
				
				if constraint[1] == "<=":
					if constraint[2] == 3232235522:
						constraint[1] = "=="
						constraint[2] = "192.168.0.2"
					
					# should not happen since it would not
					# be the best split for the node
					elif constraint[2] == 3232235523:
						constraint[1] = "=="
						constraint[2] = "192.168.0.2"
						new_rule = deepcopy(rule)
						new_rules.append(new_rule)
						constraint[2] = "192.168.0.3"

				elif constraint[1] == ">":
					if constraint[2] == 3232235522:
						constraint[1] = "=="
						constraint[2] = "192.168.0.3"

	new_rules = rules2 + new_rules
	return new_rules

# round non-integer thresholds when necessary
def round_rules(rules):

	rules2 = deepcopy(rules)

	for rule in rules2:
		for constraint in rule[0]:
			
			if constraint[1] == "<=":
				if constraint[0] != "dst":
					constraint[2] = floor(constraint[2])

			if constraint[1] == ">":
				if constraint[0] != "dst":
					constraint[2] = floor(constraint[2])

		rule[1] = int(rule[1])

	return rules2

# rewrite rules contraining constraints on pre-processed port
# numbers as a set of rules with constraints on the actual port
# number
def ports_rules_adapter(rules):

	rules2 = deepcopy(rules)
	new_rules = []
	to_remove = []

	for rule in rules2:
		for constraint in rule[0]:
			
			# should not happen (bad split)
			if "port" in constraint[0]\
				and constraint[1] == "<="\
				and constraint[2] == 1024:

				constraint[1] = ">"
				new_rule = deepcopy(rule)
				new_rules.append(new_rule)
				constraint[1] = "<="

			# should not happen (bad split)
			if "port" in constraint[0]\
				and constraint[1] == ">"\
				and constraint[2] == 1024:

				to_remove.append(rule)

	return [i for i in rules2 if i not in to_remove] + new_rules


# rewrite rules contraining constraints on pre-processed checksums
# numbers as a set of rules with constraints on the actual checksum

def checksum_rules_adapter(rules):

	rules2 = deepcopy(rules)
	new_rules = []
	to_remove = []

	for rule in rules2:
		for constraint in rule[0]:
			
			# should not happen (bad split)
			if constraint[0] == "chksum"\
				and constraint[1] == "<="\
				and constraint[2] == 1:

				constraint[1] = ">"
				new_rule = deepcopy(rule)
				new_rules.append(new_rule)
				constraint[1] = "<="

			# should not happen (bad split)
			if constraint[0] == "chksum"\
				and constraint[1] == ">"\
				and constraint[2] == 1:

				to_remove.append(rule)

	return [i for i in rules2 if i not in to_remove] + new_rules

# rewrite rules contraining constraints on the transport protocol
# so that only equality constraints are used

def proto_rules_adapter(rules):

	rules2 = deepcopy(rules)

	for rule in rules2:
		for constraint in rule[0]:

			if constraint[0] == "proto":
				if constraint[1] == "<=":
					if constraint[2] < 17:

						constraint[1] = "=="
						constraint[2] = 6

					else:
						constraint[1] = "=="
						constraint[2] = 17

				if constraint[1] == ">":

					# should be the only possibility 
					# (only values in range [6,17] considered)
					if constraint[2] >= 6:
						constraint[1] = "=="
						constraint[2] = 17

	return rules2

# rewrites rules without constraints on the transport protocol
# as a set of 2 specialized equivalent rules
def proto_rules_multiplier(rules):

	rules2 = deepcopy(rules)
	new_rules = []

	for rule in rules2:
		
		protoFound = False
		
		for constraint in rule[0]:
			if constraint[0] == "proto":
				protoFound = True

		if not protoFound:
			new_rule = deepcopy(rule)
			constraint = ["proto", "==", 6]
			rule[0].append(constraint)
			constraint2 = ["proto", "==", 17]
			new_rule[0].append(constraint2)
			new_rules.append(new_rule)

	return rules2 + new_rules

# rewrites rules containing constraints on the payload length
# as sets of rules with constraints on fields that are available in mmb
def payload_length_rule_multiplier(rules):

	rules2 = deepcopy(rules)
	new_rules = []
	to_multiply = []

	for rule in rules2:
		
		proto = 0

		for constraint in rule[0]:
			if constraint[0] == "proto":
				proto = constraint[2]

		for constraint in rule[0]:
			if constraint[0] == "payload_len":

				if proto == 17:
					constraint[0] = "udp_len"
					constraint[2] = constraint[2] + 8
			
				else:
					to_multiply.append(rule)
					break
					
	rules2 = [i for i in rules2 if i not in to_multiply]
	
	for rule in to_multiply:
		
		lower_bound = None
		upper_bound = None

		for constraint in rule[0]:
			if constraint[0] == "payload_len":
				
				if constraint[1] == "<=":
					upper_bound = constraint[2]

				elif constraint[1] == ">":
					lower_bound = constraint[2]

		for ip_ihl in range(5, 16):
			for tcp_offset in range(5, 16):

				new_lower_bound = None
				new_upper_bound = None
				
				if lower_bound is not None:
					new_lower_bound = lower_bound + 4 * (ip_ihl + tcp_offset)

				if upper_bound is not None:
					new_upper_bound = upper_bound + 4 * (ip_ihl + tcp_offset)

				new_rule = deepcopy(rule)
				new_rule = [[i for i in new_rule[0] if i[0] != "payload_len"],\
				                                                new_rule[1]]

				if new_lower_bound is not None:
					new_constraint1 = ["ip_tot_len", ">", new_lower_bound]
					new_constraint2 = ["ip_ihl", "==", ip_ihl]
					new_constraint3 = ["tcp_offset", "==", tcp_offset]
					new_rule[0].append(new_constraint1)
					new_rule[0].append(new_constraint2)
					new_rule[0].append(new_constraint3)

				if new_upper_bound is not None:
					new_constraint1 = ["ip_tot_len", "<=", new_upper_bound]
					new_constraint2 = ["ip_ihl", "==", ip_ihl]
					new_constraint3 = ["tcp_offset", "==", tcp_offset]
					new_rule[0].append(new_constraint1)
					new_rule[0].append(new_constraint2)
					new_rule[0].append(new_constraint3)

				new_rules.append(new_rule)



	return rules2 + new_rules

# write rules in a valid mmb format in the file
def get_rules_mmb_format(rules, filename):

	f = open(filename, "w")

	if not f:
		print("Could not create file")
		return

	# to remove: payload len
	tcp_aliases = {"proto": "ip-proto", "dst": "ip-daddr","dport":"tcp-dport",\
	                                    "sport": "tcp-sport", "chksum": "tcp-checksum",\
	                                    "payload_len": "ip-len", "ip_tot_len": "ip-len",\
	                                    "ip_ihl" : "ip-ihl", "tcp_offset": "tcp-offset"}

	udp_aliases = {"proto": "ip-proto", "dst": "ip-daddr","dport":"udp-dport","sport": "udp-sport",\
	                                                                          "chksum": "udp-checksum","udp_len": "udp-len",
	                                                                          "payload_len" : "ip_len"}

	for rule in rules:
		proto = [constraint for constraint in rule[0]\
		                    if constraint[0] == "proto"]
		proto = proto[0][2]
		
		if rule[1] == 1:
			f.write("vppctl \"mmb add")

		for constraint in rule[0]:
			if proto == 6 and rule[1] == 1:
				f.write(" " + tcp_aliases[constraint[0]] + " " \
					                   + constraint[1] + " " + str(constraint[2]))
			if proto == 17 and rule[1] == 1:
				f.write(" " + udp_aliases[constraint[0]] + " " \
					                   + constraint[1] + " " + str(constraint[2]))

		if rule[1] == 1:
			f.write(" drop\"\n")


# extracts rules from the tree, reverses pre-processing,
# writes them to the out.txt file
def extract_tree_rules(tree_classifier, X, features_names):

	tree = tree_classifier.tree_
	leaves_values = get_leaves_values(tree_classifier, X)

	rules = get_rules(tree, leaves_values, 0, [], features_names, [])
	rules = simplify_rules(rules)
	rules = ip_rule_adapter(rules)
	rules = round_rules(rules)
	rules = ports_rules_adapter(rules)
	rules = checksum_rules_adapter(rules)
	rules = proto_rules_adapter(rules)
	rules = proto_rules_multiplier(rules)

	rules = payload_length_rule_multiplier(rules)

	for i in rules:
	 	print(i)
	print(len(rules))

	get_rules_mmb_format(rules, "out.txt")

# get the decision tree's leaves' labels
def get_leaves_values(tree_classifier, X):
	
	node_indicator = tree_classifier.decision_path(X)
	leaves_id = tree_classifier.apply(X)
	y_pred = tree_classifier.predict(X)

	leaves_output = {}
	
	for i in range(X.shape[0]):
		
		leaf = leaves_id[i]

		if leaf not in leaves_output:
			leaves_output[leaf] = y_pred[i]


	return leaves_output

# extracts	the rules from a decision tree
# $tree is the decision tree
# $node_id must be 0 initially
# $rule_set must be [] initially
# $features_names contains the name of the different features in order
# $rules must be [] initially
def get_rules(tree, leaves_values, node_id, rule_set, features_names, rules):

	features = tree.feature
	thresholds = tree.threshold
	predict = tree.predict
	children_left = tree.children_left
	children_right = tree.children_right
	
	is_split_node = children_left[node_id] != children_right[node_id]


	if not is_split_node:
		rules.append([rule_set, leaves_values[node_id]])

	else:

		feature_name = features_names[features[node_id]]
		threshold = thresholds[node_id]
		left_child = children_left[node_id]
		right_child = children_right[node_id]

		rule_set1 = rule_set.copy()
		rule_set1.append([feature_name, "<=", threshold])
		get_rules(tree, leaves_values, left_child,rule_set1, features_names, 
															 rules)

		rule_set1 = rule_set.copy()
		rule_set1.append([feature_name, ">", threshold])
		get_rules(tree, leaves_values, right_child, rule_set1, features_names, 
															   rules)

	return rules