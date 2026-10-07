#!/usr/bin/perl
use strict;
use warnings;

package Shop::Cart;

# A shopping cart.
sub new {
    my ($class, %args) = @_;
    my $self = { items => {}, currency => $args{currency} // 'USD' };
    return bless $self, $class;
}

# Adds an item.
sub add {
    my ($self, $sku, $qty) = @_;
    $self->{items}{$sku} += $qty // 1;
    return $self;
}

sub total {
    my ($self, $prices) = @_;
    my $total = 0;
    for my $sku (keys %{ $self->{items} }) {
        $total += ($prices->{$sku} // 0) * $self->{items}{$sku};
    }
    return $total;
}

package main;

my $cart = Shop::Cart->new(currency => 'EUR');
$cart->add('apple', 2)->add('pear');
printf "%.2f\n", $cart->total({ apple => 0.5, pear => 0.75 });

my @sorted = sort { $a <=> $b } (3, 1, 2);
print "@sorted\n" if @sorted;

__END__

=head1 NAME

Shop::Cart - a cart

=cut
