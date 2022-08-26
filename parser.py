import argparse

class KParseArgs():

    def __init__(self):
        self.args = parser = argparse.ArgumentParser()

        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--model_type", help="model type", action='store', nargs='?', default="base_line",
                            type=str)

        self.parser.add_argument("--opt", help="optimizer", action='store', nargs='?', default="adam",
                            type=str)
        self.parser.add_argument("--batch_size", help="batch_size", action='store', nargs='?', default="32",
                            type=int)

        self.parser.add_argument("--save_model", help="save_model", action='store', nargs='?', default="1",
                            type=int)
                            
                            
                            
        self.parser.add_argument("--embedding_dim", help="embedding size", action='store', nargs='?', default=50,
                            type=int)
        
        self.parser.add_argument("--max_len", help="maximum length", action='store', nargs='?', default=50,
                            type=int)
        self.parser.add_argument("--num_classes", help="number of classes", action='store', nargs='?', default=4,
                            type=int)
        self.parser.add_argument("--epochs", help="Number of epochs for training", nargs='?', action='store', default=20,
                            type=int)
        self.parser.add_argument("--loss", help="Loss Function for the Gradients", nargs='?', action='store',
                            default='categorical_crossentropy', type=str)
        self.parser.add_argument("--load_model_path", help="Load model path", nargs='?', action='store', default='/tmp', type=str)
        self.parser.add_argument("--my_review", help="Type in your review", nargs='?', action='store', default='this film was horrible, bad acting, even worse direction', type=str)
        self.parser.add_argument("--verbose", help="Verbose output", nargs='?', action='store', default=0, type=int)
    
    def parse_args(self):
        return self.parser.parse_args()

    def parse_args_list(self, args_list):
        return self.parser.parse_args(args_list)
        


