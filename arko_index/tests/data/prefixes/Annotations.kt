@file:JvmName("Repos")

package shop.repos

import javax.inject.Inject

class Repo {
    val db: Db
    @Inject
    lateinit var cache: Cache

    var x: Int = 0
        @JvmName("getX") get
        @Deprecated("a b") set

    val size: Int
        @Suppress("x")
        get() = db.count()
}

class Plain
@Suppress("unused")
class Next @Inject constructor(val repo: Repo)

val total = 1
@Deprecated("old")
fun legacy() = total
