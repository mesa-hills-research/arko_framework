{-# LANGUAGE ScopedTypeVariables #-}
-- | A small stack machine.
module Machine
  ( Instr (..)
  , run
  , Stack
  ) where

import qualified Data.Map as Map
import Data.List (foldl')

-- | The machine's instructions.
data Instr
  = Push Int
  | Add
  | Mul
  | Dup
  deriving (Show, Eq)

type Stack = [Int]

newtype Program = Program [Instr]

class Pretty a where
  pretty :: a -> String

instance Pretty Instr where
  pretty (Push n) = "push " ++ show n
  pretty Add = "add"
  pretty Mul = "mul"
  pretty Dup = "dup"

-- | Runs one instruction.
step :: Stack -> Instr -> Either String Stack
step s (Push n) = Right (n : s)
step (a : b : s) Add = Right (a + b : s)
step (a : b : s) Mul = Right (a * b : s)
step (a : s) Dup = Right (a : a : s)
step _ i = Left ("stack underflow at " ++ pretty i)

-- | Runs a program from an empty stack.
run :: [Instr] -> Either String Stack
run = foldl' go (Right [])
  where
    go acc i = acc >>= \s -> step s i

main :: IO ()
main = do
  let program = [Push 2, Push 3, Add, Dup, Mul]
  print (run program)
