# Summary statistics for a data frame.
library(dplyr)

#' The coefficient of variation.
#' @param x A numeric vector.
cv <- function(x, na.rm = TRUE) {
  sd(x, na.rm = na.rm) / mean(x, na.rm = na.rm)
}

summarize_column <- function(df, column) {
  values <- df[[column]]
  list(
    mean = mean(values),
    cv = cv(values),
    range = range(values)
  )
}

# An S4 class for measurements.
setClass("Measurement", representation(value = "numeric", unit = "character"))

setGeneric("convert", function(m, unit) standardGeneric("convert"))

setMethod("convert", "Measurement", function(m, unit) {
  if (m@unit == unit) return(m)
  stop("unsupported conversion")
})

square = function(x) x^2

results <- lapply(names(mtcars), function(col) summarize_column(mtcars, col))

for (i in seq_along(results)) {
  if (results[[i]]$cv > 0.5) {
    message("high variation in ", names(mtcars)[i])
  }
}

mtcars %>%
  group_by(cyl) %>%
  summarise(mpg = mean(mpg))
